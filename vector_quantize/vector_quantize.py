import math
from copy import deepcopy
from functools import partial
from math import prod
from typing import Callable

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import HeUniform, initializer
from mindspore.communication import get_rank

from .utils import einsum_ms

#### Note: 关于分布式训练：vq作为一个运算步骤不再单独设计训练代码，也不再单独做init, set_auto_parallel_context等分布式训练配置，
#### 因此，在VQ中加入了distribute参数用于手动指引当前是否处于分布式环境下


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def identity(t):
    return t


def l2norm(t):
    return ops.L2Normalize(-1)(t)


def cdist(x, y):
    x2 = (x**2).sum(-1)
    y2 = (y**2).sum(-1)
    xy = einsum_ms("b i d, b j d -> b i j", x, y) * -2
    return (x2.expand_dims(-1) + y2.expand_dims(1) + xy).sqrt()


def log(t, eps=1e-20):
    return ops.log(t.clamp(min=eps))


def ema_inplace(old, new, decay):
    old = old.lerp(new, 1 - decay)


def uniform_init(*shape):
    t = Parameter(initializer(HeUniform(), shape)).value()
    return t


def gumbel_noise(t):
    noise = ops.uniform(t.shape, Tensor(0, dtype=ms.float32), Tensor(1, dtype=ms.float32))
    return -log(-log(noise))


def gumbel_sample(
    logits, temperature=1.0, stochastic=False, straight_through=False, reinmax=False, dim=-1, training=True
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    # for each input point, find the index of the codeword that has the farthest distance to the input point
    # and locate the codewords(one-hot did it)
    ind = sampling_logits.argmax(axis=dim)
    one_hot = ops.one_hot(ind, size).to(dtype)

    assert not (
        reinmax and not straight_through
    ), "reinmax can only be turned on if using straight through gumbel softmax"

    if not straight_through or temperature <= 0.0 or not training:
        return ind, one_hot

    # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
    # algorithm 2

    if reinmax:
        π0 = ops.softmax(logits, axis=dim)
        π1 = (one_hot + ops.softmax(logits / temperature,axis=dim)) / 2
        π1 = ops.softmax(Tensor(log(π1) - logits) + logits,axis=1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - Tensor(π2) + one_hot
    else:
        π1 = ops.softmax(logits / temperature,axis=dim)
        one_hot = one_hot + π1 - Tensor(π1)

    return ind, one_hot


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(axis=dim, keepdims=True)
    return (x + eps) / (denom + n_categories * eps)


def sample_vectors(samples, num):
    num_samples = samples.shape[0]
    if num_samples >= num:
        indices = ops.randperm(num_samples)[:num]
    else:
        indices = ops.randint(0, num_samples, (num,))

    return samples[indices]


def batched_sample_vectors(samples, num):
    return ops.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], axis=0)


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    total_count = ops.full((), total_count, dtype=probs.dtype)
    remainder = probs.new_ones(())
    sample = ms.numpy.empty_like(probs, dtype=ms.int64)

    for i, p in enumerate(probs):
        s = Tensor(np.random.binomial(total_count, p / remainder))
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample


def all_gather_sizes(x, dim):
    size = Tensor(x.shape[dim], dtype=ms.int64)
    all_sizes = ops.AllGather()(size)
    return ops.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else ms.numpy.empty(shape=pad_shape(x.shape, size, dim), dtype=x.dtype)
        t = ops.Broadcast(root_rank=i)((t,))
        all_x.append(t[0])

    return all_x


def sample_vectors_distributed(local_samples, num):
    local_samples = local_samples[0]

    rank = get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = ms.numpy.empty_like(all_num_samples)

    samples_per_rank = ops.Broadcast(root_rank=0)((samples_per_rank,))[0]
    samples_per_rank = samples_per_rank.asnumpy().tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = ops.cat(all_samples, axis=0)

    return out.expand_dims(0)


def batched_bincount(x, *, minlength):
    """x must be a 2-dim tensor here"""
    batch, dtype = x.shape[0], x.dtype
    target = ops.zeros((batch, minlength), dtype=dtype)
    values = ops.ones_like(x)
    indices = ops.cat(
        ((ops.ones(x.shape, dtype=x.dtype) * ops.arange(batch).expand_dims(-1)).expand_dims(-1), x.expand_dims(-1)),
        axis=-1,
    )
    target = ops.tensor_scatter_add(target, indices, values)
    return target


def kmeans(
    samples, num_clusters, num_iters=10, use_cosine_sim=False, sample_fn=batched_sample_vectors, all_reduce_fn=noop
):
    num_codebooks, dim, dtype = samples.shape[0], samples.shape[-1], samples.dtype

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.movedim(-1, 1)
        else:
            dists = -cdist(samples, means)

        buckets = ops.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        if all_reduce_fn != noop:
            bins = all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros((num_codebooks, num_clusters, dim), dtype=dtype)
        indices = buckets.expand_dims(-1).repeat(dim, -1)
        point_indices = ops.cat(
            (
                (
                    ops.ones(indices.shape, dtype=indices.dtype) * ops.arange(indices.shape[0]).expand_dims(-1)
                ).expand_dims(-1),
                indices.expand_dims(-1),
                ops.ones(indices.shape, dtype=indices.dtype).expand_dims(-1)
                * ops.arange(indices.shape[-1]).expand_dims(-1),
            ),
            axis=-1,
        )
        new_means = ops.tensor_scatter_add(new_means, point_indices, samples)
        new_means = new_means / (bins_min_clamped.expand_dims(-1))
        if all_reduce_fn != noop:
            new_means = all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = ops.where(zero_mask.expand_dims(-1), means, new_means)

    return means, bins


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = indices.expand_dims(-1).repeat(dim, -1)
    embeds = embeds.expand_dims(1).repeat(batch, 1)
    return embeds.gather_elements(2, indices)


# regularization losses


def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum_ms("h i d, h j d -> h i j", normed_codes, normed_codes)
    return (cosine_sim**2).sum() / (h * n**2) - (1 / n)


# distance types


class EuclideanCodebook(nn.Cell):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        reset_cluster_size=None,
        learnable_codebook=False,
        gumbel_sample=gumbel_sample,
        sample_codebook_temp=1.0,
        ema_update=True,
        affine_param=False,
        sync_affine_param=False,
        affine_param_batch_decay=0.99,
        affine_param_codebook_decay=0.9,
        distribute=False,
    ):
        super().__init__()
        self.transform_input = identity

        self.decay = decay
        self.ema_update = ema_update

        init_fn = uniform_init if not kmeans_init else ops.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        assert not (
            distribute and num_codebooks > 1 and kmeans_init
        ), "kmeans init is not compatible with multiple codebooks in distributed environment for now"

        self.sample_fn = sample_vectors_distributed if distribute and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = ops.AllReduce() if distribute and sync_kmeans else noop
        self.all_reduce_fn = ops.AllReduce() if distribute else noop

        self.initted = Tensor([not kmeans_init])
        self.cluster_size = ops.zeros((num_codebooks, codebook_size))
        self.embed_avg = deepcopy(embed)

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = Parameter(embed)
        else:
            self.embed = Parameter(embed, requires_grad=False)

        # affine related params

        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            return

        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay

        self.batch_mean = None
        self.batch_variance = None

        self.codebook_mean_needs_init = Tensor([True])
        self.codebook_mean = ms.numpy.empty((num_codebooks, 1, dim))
        self.codebook_variance_needs_init = Tensor([True])
        self.codebook_variance = ms.numpy.empty((num_codebooks, 1, dim))

    def init_embed_(self, data, mask=None):
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data_shape = data[mask].shape
            data = data.reshape(c, int(data_shape[0] / c), data_shape[-1])

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        embed_sum = embed * cluster_size.expand_dims(-1)

        self.embed.assign_value(embed)
        self.embed_avg.assign_value(embed_sum)
        self.cluster_size.assign_value(cluster_size)
        self.initted = Tensor([True])

    def update_affine(self, data, embed, mask=None):
        assert self.affine_param

        # calculate codebook mean and variance
        embed_shape = embed.shape
        embed = embed.reshape(
            (embed_shape[0], int(math.prod(embed_shape) / embed_shape[0] / embed_shape[-1]), embed_shape[-1])
        )

        if self.training:
            self.codebook_mean = embed.mean(1, keep_dims=True)
            self.codebook_mean_needs_init = Tensor([False])
            self.codebook_variance = embed.var(1, ddof=False, keepdims=True)
            self.codebook_variance_needs_init = Tensor([False])

        # prepare batch data, which depends on whether it has masking

        data_shape = data.shape
        data = data.reshape(
            (data_shape[0], int(math.prod(data_shape) / data_shape[0] / data_shape[-1]), data_shape[-1])
        )

        if exists(mask):
            c = data.shape[0]
            data_shape = data.shape
            data = data.reshape((c, int(data_shape[0] / c)) + (data_shape[-1],))

        # calculate batch mean and variance

        if not self.sync_affine_param:
            self.batch_mean = data.mean(1, keep_dims=True)
            self.batch_variance = data.var(1, ddof=False, keepdims=True)
            return

        num_vectors, dtype = data.shape[-2], data.dtype

        # number of vectors, for denominator

        num_vectors = Tensor([num_vectors], dtype=dtype)
        num_vectors = ops.AllReduce()(num_vectors)

        # calculate distributed mean

        batch_sum = data.sum(1, keepdims=True)
        batch_sum = ops.AllReduce()(batch_sum)
        batch_mean = batch_sum / num_vectors

        if not self.sync_affine_param:
            self.batch_mean = self.batch_mean * self.affine_param_batch_decay + Tensor(batch_mean) * (
                1 - self.affine_param_batch_decay
            )
        else:
            self.batch_mean = Tensor(batch_mean)

        # calculate distributed variance

        variance_numer = ((data - batch_mean) ** 2).sum(1, keepdims=True)
        variance_numer = ops.AllReduce()(variance_numer)
        batch_variance = variance_numer / num_vectors

        if not self.sync_affine_param:
            self.batch_variance = self.batch_variance * self.affine_param_batch_decay + Tensor(batch_variance) * (
                1 - self.affine_param_batch_decay
            )
        else:
            self.batch_variance = Tensor(batch_variance)

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))):
            if not ops.any(mask):
                continue

            sampled = self.sample_fn(samples.expand_dims(0), mask.sum().item())
            sampled = sampled[0]

            self.embed.data[ind][mask] = sampled

            self.cluster_size.value()[ind][mask] = self.reset_cluster_size
            self.embed_avg.value()[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not ops.any(expired_codes):
            return

        batch_samples_shape = batch_samples.shape
        batch_samples = batch_samples.reshape(
            (batch_samples_shape[0], int(math.prod(batch_samples_shape[1:][:-1])), batch_samples_shape[-1])
        )
        self.replace(batch_samples, batch_mask=expired_codes)

    def construct(self, x, sample_codebook_temp=None, mask=None, freeze_codebook=False):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = x.expand_dims(0)

        # transform input to a 3-dim tensor
        x_shape = x.shape
        flatten = x.reshape((x_shape[0], prod(x_shape[1:][:-1]), x_shape[-1]))

        if exists(mask):
            mask = mask.expand_dims(0).repeat(flatten.shape[0], 0).expand_dims(-2).repeat(flatten.shape[-2] // (mask.shape[0] * mask.shape[1]), -2)
            mask = mask.reshape(
                (mask.shape[0], math.prod(mask.shape[1:]))
            )

        # the "embed" is the codebook with sahpe (num_codebooks, codebook_size, dim)
        # the "flatten" is the shaped input
        # if kmeans_init=True, here the function will initialize codewords with k-means
        # (which means initial codewords are choosing from input points),
        # if kmeans_init=False, at the very first, in the __init__ function, codewords will be initialized
        # to randdom numbers using uniform distribution
        self.init_embed_(flatten, mask=mask)

        if self.affine_param:
            self.update_affine(flatten, self.embed, mask=mask)

        embed = self.embed if self.learnable_codebook else self.embed.value()

        if self.affine_param:
            codebook_std = self.codebook_variance.clamp(min=1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min=1e-5).sqrt()
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

        # calculate the distance of input and each codeword,
        # dist.shape[-1] = embed.shape[-1], is just the number of codewords
        dist = -cdist(flatten, embed)

        # for each input point get the index of the worst codewords, embed_onehot is the one-hot version of embed_ind
        embed_ind, embed_onehot = self.gumbel_sample(
            dist, dim=-1, temperature=sample_codebook_temp, training=self.training
        )

        embed_ind = embed_ind.reshape(embed_ind.shape[:1] + x_shape[1:][:-1])

        # for each input point, get the corresponding codeword according to onehot or index,
        # TODO: but not sure about why 2 branches here
        if self.training:
            # get the specific value of the worst codeword according to onehot
            unpacked_onehot = embed_onehot.reshape(embed_onehot.shape[:1] + x_shape[1:][:-1] + embed_onehot.shape[-1:])
            # for each batch(h), matrix(b n c) is multipled by matrix(c d), so we can get a matrix(h b n d) ultimately
            quantize = einsum_ms("h b n c, h c d -> h b n d", unpacked_onehot, embed)
        else:
            # for each input point, get the specific value of the worst codeword according to index
            # (thus have the same shape as input)
            quantize = batched_embedding(embed_ind, embed)

        if self.training and self.ema_update and not freeze_codebook:
            if self.affine_param:
                flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean

            if exists(mask):
                embed_onehot[~mask] = 0.0

            cluster_size = embed_onehot.sum(axis=1)

            if self.all_reduce_fn != noop:
                cluster_size = self.all_reduce_fn(cluster_size)
            self.cluster_size.assign_value(self.cluster_size.lerp(cluster_size, 1 - self.decay))

            embed_sum = einsum_ms("h n d, h n c -> h c d", flatten, embed_onehot)
            if self.all_reduce_fn != noop:
                embed_sum_ = deepcopy(embed_sum)
                embed_sum = self.all_reduce_fn(embed_sum_)
            self.embed_avg.assign_value(self.embed_avg.lerp(embed_sum, 1 - self.decay))

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(
                axis=-1, keepdims=True
            )

            embed_normalized = self.embed_avg / cluster_size.expand_dims(-1)
            self.embed.assign_value(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: t[0], (quantize, embed_ind))

        dist = dist.reshape(dist.shape[:1] + x_shape[1:][:-1] + dist.shape[-1:])

        return quantize, embed_ind, dist


class CosineSimCodebook(nn.Cell):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        reset_cluster_size=None,
        learnable_codebook=False,
        gumbel_sample=gumbel_sample,
        sample_codebook_temp=1.0,
        ema_update=True,
        distribute=False,
    ):
        super().__init__()
        self.transform_input = l2norm

        self.ema_update = ema_update
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = ops.zeros((num_codebooks, codebook_size, dim))

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if distribute and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = ops.AllReduce() if distribute and sync_kmeans else noop
        self.all_reduce_fn = ops.AllReduce() if distribute else noop

        self.initted = Tensor([not kmeans_init])
        self.cluster_size = ops.zeros((num_codebooks, codebook_size))
        self.embed_avg = deepcopy(embed)

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = Parameter(embed)
        else:
            self.embed = embed

    def init_embed_(self, data, mask=None):
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data_shape = data.shape
            data = data.reshape(c, int(data_shape[0] / c), data_shape[-1])

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim=True,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        embed_sum = embed * cluster_size.expand_dims(-1)

        self.embed.assign_value(embed)
        self.embed_avg.assign_value(embed_sum)
        self.cluster_size.assign_value(cluster_size)
        self.initted = Tensor([True])

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))):
            if not ops.any(mask):
                continue

            sampled = self.sample_fn(samples.expand_dims(0), mask.sum().item())
            sampled = sampled[0]

            self.embed.data[ind][mask] = sampled
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size
            self.cluster_size.data[ind][mask] = self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not ops.any(expired_codes):
            return

        batch_samples_shape = batch_samples.shape
        batch_samples = batch_samples.reshape(
            (batch_samples_shape[0], int(math.prod(batch_samples_shape[1:][:-1])), batch_samples_shape[-1])
        )
        self.replace(batch_samples, batch_mask=expired_codes)

    def construct(self, x, sample_codebook_temp=None, mask=None, freeze_codebook=False):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = x.expand_dims(0)

        x_shape = x.shape
        flatten = x.reshape((x_shape[0], prod(x_shape[1:][:-1]), x_shape[-1]))

        if exists(mask):
            mask = mask.expand_dims(0).repeat(flatten.shape[0], 0).expand_dims(-2).repeat(flatten.shape[-2] // (mask.shape[0] * mask.shape[1]), -2)
            mask = mask.reshape(
                (mask.shape[0], math.prod(mask.shape[1:]))
            )

        self.init_embed_(flatten, mask=mask)

        embed = self.embed if self.learnable_codebook else self.embed.value()

        dist = einsum_ms("h n d, h c d -> h n c", flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(
            dist, dim=-1, temperature=sample_codebook_temp, training=self.training
        )
        embed_ind = embed_ind.reshape(embed_ind.shape[:1] + x_shape[1:][:-1])

        if self.training:
            unpacked_onehot = embed_onehot.reshape(embed_onehot.shape[:1] + x_shape[1:][:-1] + embed_onehot.shape[-1:])
            quantize = einsum_ms("h b n c, h c d -> h b n d", unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)

        if self.training and self.ema_update and not freeze_codebook:
            if exists(mask):
                embed_onehot[~mask] = 0.0

            bins = embed_onehot.sum(axis=1)
            if self.all_reduce_fn != noop:
                bins = self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size.data, bins, self.decay)

            embed_sum = einsum_ms("h n d, h n c -> h c d", flatten, embed_onehot)
            embed_sum_ = deepcopy(embed_sum)
            embed_sum = self.all_reduce_fn(embed_sum_)
            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(
                axis=-1, keepdims=True
            )

            embed_normalized = self.embed_avg / cluster_size.expand_dims(-1)
            embed_normalized = l2norm(embed_normalized)

            self.embed.assign_value(l2norm(embed_normalized))
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: t[0], (quantize, embed_ind))

        dist = dist.reshape(dist.shape[:1] + x_shape[1:][:-1] + dist.shape[-1:])
        return quantize, embed_ind, dist


# main class


class VectorQuantize(nn.Cell):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim=None,
        heads=1,
        separate_codebook_per_head=False,
        decay=0.8,
        eps=1e-5,
        freeze_codebook=False,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        use_cosine_sim=False,
        threshold_ema_dead_code=0,
        channel_last=True,
        accept_image_fmap=False,
        commitment_weight=1.0,
        commitment_use_cross_entropy_loss=False,
        orthogonal_reg_weight=0.0,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
        stochastic_sample_codes=False,
        sample_codebook_temp=1.0,
        straight_through=False,
        reinmax=False,  # using reinmax for improved straight-through, assuming straight through helps at all
        sync_codebook=None,
        distribute=False,
        sync_affine_param=False,
        ema_update=True,
        learnable_codebook=False,
        in_place_codebook_optimizer: Callable[..., nn.Optimizer] = None,
        # Optimizer used to update the codebook embedding if using learnable_codebook
        affine_param=False,
        affine_param_batch_decay=0.99,
        affine_param_codebook_decay=0.9,
        sync_update_v=0.0
        # the v that controls optimistic vs pessimistic update for synchronous update rule (21)
        # https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Dense(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Dense(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.has_projections = requires_projection

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = (
            commitment_use_cross_entropy_loss  # whether to use cross entropy loss to codebook as commitment loss
        )

        self.learnable_codebook = learnable_codebook

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        assert not (ema_update and learnable_codebook), "learnable codebook not compatible with EMA update"

        assert 0 <= sync_update_v <= 1.0
        assert not (sync_update_v > 0.0 and not learnable_codebook), "learnable codebook must be turned on"

        self.sync_update_v = sync_update_v

        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        gumbel_sample_fn = partial(
            gumbel_sample, stochastic=stochastic_sample_codes, reinmax=reinmax, straight_through=straight_through
        )

        if not exists(sync_codebook):
            sync_codebook = distribute

        codebook_kwargs = dict(
            dim=codebook_dim,
            num_codebooks=heads if separate_codebook_per_head else 1,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sync_kmeans=sync_kmeans,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            distribute=sync_codebook,
            learnable_codebook=has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp=sample_codebook_temp,
            gumbel_sample=gumbel_sample_fn,
            ema_update=ema_update,
        )

        if affine_param:
            assert not use_cosine_sim, "affine param is only compatible with euclidean codebook"
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param=True,
                sync_affine_param=sync_affine_param,
                affine_param_batch_decay=affine_param_batch_decay,
                affine_param_codebook_decay=affine_param_codebook_decay,
            )

        self._codebook = codebook_class(**codebook_kwargs).to_float(ms.float32)

        self.in_place_codebook_optimizer = (
            in_place_codebook_optimizer(self._codebook.trainable_params())
            if exists(in_place_codebook_optimizer)
            else None
        )

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return codebook[0]

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = codes.expand_dims(0)

        self._codebook.embed.assign_value(codes)

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            codes_shape = codes.shape
            return codes.reshape(codes_shape[:-2] + (codes_shape[-1] * codes_shape[-2],))

        indices_shape = indices.shape
        indices = indices.reshape((indices_shape[0], prod(indices_shape[1:][:-1]), indices_shape[-1]))
        indices = indices.movedim(-1, 1)

        indices = indices.expand_dims(-1).repeat(codebook.shape[-1], -1)
        codebook = codebook.expand_dims(0).repeat(indices.shape[0], 0)

        codes = codebook.gather_elements(2, indices)
        codes = codes.movedim(2, 1)
        codes_shape = codes.shape
        codes = codes.reshape(codes_shape[:-2] + (codes_shape[-1] * codes_shape[-2],))
        codes = codes.reshape(codes.shape[:1] + indices_shape[1:][:-1] + codes.shape[-1:])
        return codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        return self.project_out(codes)

    def construct(self, x, indices=None, mask=None, sample_codebook_temp=None, freeze_codebook=False):
        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert not exists(mask)
            x = x.expand_dims(1)
        shape, heads, is_multiheaded, _, return_loss = (
            x.shape,
            self.heads,
            self.heads > 1,
            self.codebook_size,
            exists(indices),
        )

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        # rearrange inputs

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x_shape = x.shape
            assert len(x_shape) == 4, f"expected 4 dims. Received {len(x_shape)}-dim tensor."
            x = x.reshape(x_shape[:-2] + (x_shape[-1] * x_shape[-2],))
            x = x.movedim(1, -1)

        if need_transpose:
            x = x.movedim(-1, 1)

        # project input

        x = self.project_in(x)

        # handle multi-headed separate codebooks

        if is_multiheaded:
            if self.separate_codebook_per_head:
                x_shape = x.shape
                x = x.reshape((x_shape[:2] + (heads, int(x_shape[-1] / heads))))
                x = x.movedim(2, 0)
            else:
                x_shape = x.shape
                x = x.reshape((x_shape[:2] + (heads, int(x_shape[-1] / heads))))
                x = x.movedim(1, 2)
                x_shape = x.shape
                x = x.reshape((int(math.prod(x_shape[:2])),) + x_shape[2:])
                x = x.expand_dims(0)

        # l2norm for cosine sim, otherwise identity

        x = self._codebook.transform_input(x)

        # codebook forward kwargs

        codebook_forward_kwargs = dict(
            sample_codebook_temp=sample_codebook_temp, mask=mask, freeze_codebook=freeze_codebook
        )

        if should_inplace_optimize and self.training and not freeze_codebook:
            # one step in-place update

            def forward_fn(x):
                quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)
                if exists(mask):
                    loss = nn.MSELoss(reduction="none")(quantize, x)
                    loss_mask = mask
                    if is_multiheaded:
                        loss_mask = mask.expand_dims(0).repeat(loss.shape[0], 0)
                        loss_mask = loss_mask.repeat(loss.shape[1] // mask.shape[0],1)

                    loss = loss[loss_mask].mean()

                else:
                    loss = nn.MSELoss()(quantize, x)
                return loss, quantize

            grad_fn = ms.value_and_grad(forward_fn, None, self.in_place_codebook_optimizer.parameters, has_aux=True)
            (loss, _), params_gradient = grad_fn(x)
            self.in_place_codebook_optimizer(params_gradient)

            # quantize again
            quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        else:
            # quantize only, no updates
            quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        if self.training:
            # determine code to use for commitment loss
            commit_quantize = Tensor(quantize) if not self.learnable_codebook or freeze_codebook else quantize

            # straight through

            quantize = x + Tensor(quantize - x)

            if self.sync_update_v > 0.0:
                # (21) in https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
                quantize = quantize + self.sync_update_v * (quantize - Tensor(quantize))

        # function for calculating cross entropy loss to distance matrix
        # used for (1) naturalspeech2 training residual vq latents to be close to the correct codes and
        # (2) cross-entropy based commitment loss

        def calculate_ce_loss(codes, distances):
            if not is_multiheaded:
                distances = distances.movedim(1, 2)
                distances = distances.expand_dims(0)
            elif self.separate_codebook_per_head:
                distances = distances.movedim(0, -1)
                distances = distances.movedim(1, 2)
            else:
                distances = distances[0]
                distances_shape = distances.shape
                distances = distances.reshape((shape[0], int(distances_shape[0] / shape[0])) + distances_shape[1:])
                distances = distances.movedim(1, -1).movedim(1, 2)

            ce_loss = ops.cross_entropy(distances, codes, ignore_index=-1)

            return ce_loss

        # if returning cross entropy loss on codes that were passed in

        if return_loss:
            return quantize, calculate_ce_loss(indices, distances)

        # transform embedding indices

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = embed_ind.movedim(0, -1)
            else:
                embed_ind = embed_ind[0]
                embed_ind_shape = embed_ind.shape
                embed_ind = embed_ind.reshape((int(embed_ind_shape[0] / heads), heads, embed_ind_shape[-1]))
                embed_ind = embed_ind.movedim(1, 2)

        if self.accept_image_fmap:
            embed_ind_shape = embed_ind.shape
            embed_ind = embed_ind.reshape((embed_ind_shape[0], height, width) + embed_ind_shape[2:])

        if only_one:
            embed_ind = embed_ind.expand_dims(1)

        # aggregate loss

        loss = Tensor([0.0])

        if self.training:
            if self.commitment_weight > 0:
                if self.commitment_use_cross_entropy_loss:
                    if exists(mask):
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = ce_loss_mask.expand_dims(-1).repeat(heads, -1)

                        embed_ind = embed_ind.masked_fill(~ce_loss_mask, -1)

                    commit_loss = calculate_ce_loss(embed_ind, distances)
                else:
                    if exists(mask):
                        # with variable lengthed sequences
                        commit_loss = ops.mse_loss(commit_quantize, x, reduction="none")

                        loss_mask = mask
                        if is_multiheaded:
                            loss_mask = loss_mask.expand_dims(0).repeat(commit_loss.shape[0], 0)
                            loss_mask = loss_mask.repeat(commit_loss.shape[1] // mask.shape[0],1)

                        commit_loss = commit_loss[loss_mask].mean()
                    else:
                        commit_loss = ops.mse_loss(commit_quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed

                # only calculate orthogonal loss for the activated codes for this batch

                if self.orthogonal_reg_active_codes_only:
                    assert not (
                        is_multiheaded and self.separate_codebook_per_head
                    ), "orthogonal regularization for only active codes not compatible with multi-headed with separate codebooks yet"
                    unique_code_ids = ops.unique(embed_ind)[0].sort()[0]
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]

                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = ops.randperm(num_codes)[: self.orthogonal_reg_max_codes]
                    codebook = codebook[:, rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        # handle multi-headed quantized embeddings

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = quantize.movedim(0, 2)
                quantize_shape = quantize.shape
                quantize = quantize.reshape(quantize_shape[:2] + (int(math.prod(quantize_shape[2:])),))
            else:
                quantize = quantize[0]
                quantize_shape = quantize.shape
                quantize = quantize.reshape((int(quantize_shape[0] / heads), heads) + quantize_shape[1:])
                quantize = quantize.movedim(1, 2)
                quantize_shape = quantize.shape
                quantize = quantize.reshape(quantize_shape[:2] + (prod(quantize_shape[2:]),))

        # project out

        quantize = self.project_out(quantize)

        # rearrange quantized embeddings

        if need_transpose:
            quantize = quantize.movedim(-1, 1)

        if self.accept_image_fmap:
            quantize = quantize.movedim(-1, 1)
            quantize_shape = quantize.shape
            quantize = quantize.reshape(quantize_shape[:2] + (height, width))

        if only_one:
            quantize = quantize.expand_dims(1)

        # if masking, only return quantized for where mask has True

        if exists(mask):
            quantize = ops.where(mask.expand_dims(-1), quantize, orig_input)

        return quantize, embed_ind, loss
