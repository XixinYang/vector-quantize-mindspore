"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

from collections import namedtuple
from math import ceil, log2, prod

from mindspore import Tensor, nn, ops
from mindspore.nn import Cell

from .utils import einsum_ms

# constants

Return = namedtuple("Return", ["quantized", "indices", "entropy_aux_loss"])

LossBreakdown = namedtuple("LossBreakdown", ["per_sample_entropy", "batch_entropy", "commitment"])


# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


# entropy


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(axis=-1)


# class


class LFQ(Cell):
    def __init__(
        self,
        *,
        dim=None,
        codebook_size=None,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        diversity_gamma=1.0,
        straight_through_activation=nn.Identity(),
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        codebook_scale=1.0,  # for residual LFQ, codebook scaled down by 2x at each layer
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(codebook_size), "either dim or codebook_size must be specified for LFQ"
        assert (
            not exists(codebook_size) or log2(codebook_size).is_integer()
        ), f"your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})"

        codebook_size = default(codebook_size, lambda: 2**dim)
        codebook_dim = int(log2(codebook_size))

        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.project_in = nn.Dense(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Dense(codebook_dims, dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # straight through activation

        self.activation = straight_through_activation

        # entropy aux loss related weights

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # for no auxiliary loss, during inference

        self.mask = 2 ** ops.arange(codebook_dim - 1, -1, -1)

        # codes

        all_codes = ops.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.codebook = codebook

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(self, indices, project_out=True):
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        if not self.keep_num_codebooks_dim:
            indices = indices.expand_dims(-1)

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)
        codes_shape = codes.shape
        codes = codes.reshape(codes_shape[:-2] + (codes_shape[-1] * codes_shape[-2],))

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if is_img_or_video:
            codes = codes.movedim(-1, 1)

        return codes

    def construct(
        self,
        x,
        inv_temperature=100.0,
        return_loss_breakdown=False,
        mask=None,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        global x_shape0
        is_img_or_video = x.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            x = x.movedim(1, -1)
            x_shape0 = x.shape
            x = x.reshape((x_shape0[0], int(prod(x_shape0[1:][:-1])), x_shape0[-1]))

        assert x.shape[-1] == self.dim, f"expected dimension of {self.dim} but received {x.shape[-1]}"

        x = self.project_in(x)

        # split out number of codebooks

        x_shape = x.shape
        x = x.reshape(x_shape[:2] + (self.num_codebooks, int(x_shape[-1] / self.num_codebooks)))

        # quantize by eq 3.

        original_input = x

        codebook_value = ops.ones_like(x) * self.codebook_scale
        quantized = ops.where(x > 0, codebook_value, -codebook_value)

        # use straight-through gradients (optionally with custom activation fn) if training

        if self.training:
            x = self.activation(x)
            x = x + Tensor(quantized - x)
        else:
            x = quantized

        # calculate indices

        indices = ((x > 0).int() * self.mask.int()).sum(-1)

        # entropy aux loss

        if self.training:
            # the same as euclidean distance up to a constant
            distance = -2 * einsum_ms("... i d, j d -> ... i j", original_input, self.codebook)

            prob = ops.softmax(-distance * inv_temperature, axis=-1)

            per_sample_entropy = entropy(prob).mean()

            # account for mask

            if exists(mask):
                prob = prob[mask]

            # distribution over all available tokens in the batch

            avg_prob = prob
            for i in range(len(prob.shape) - 2):
                avg_prob = avg_prob.mean(i)
            codebook_entropy = entropy(avg_prob).mean()

            # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
            # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

            entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
        else:
            # if not training, just return dummy 0
            entropy_aux_loss = per_sample_entropy = codebook_entropy = Tensor(0.0)

        # commit loss

        if self.training:
            commit_loss = ops.mse_loss(original_input, Tensor(quantized), reduction="none")

            if exists(mask):
                commit_loss = commit_loss[mask]

            commit_loss = commit_loss.mean()
        else:
            commit_loss = Tensor(0.0)

        # merge back codebook dim

        x_shape = x.shape
        x = x.reshape(x_shape[:-2] + (x_shape[-1] * x_shape[-2],))

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if is_img_or_video:
            x = x.reshape((x.shape[0],) + x_shape0[1:][:-1] + (x.shape[-1],))
            x = x.movedim(-1, 1)

            indices = indices.reshape(indices.shape[:1] + x_shape0[1:][:-1] + indices.shape[-1:])

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            indices = indices.squeeze(-1)

        # complete aux loss

        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

        ret = Return(x, indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)
