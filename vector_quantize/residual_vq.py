import random
from functools import partial
from itertools import zip_longest
from math import ceil, prod

from vector_quantize.vector_quantize import VectorQuantize

import mindspore as ms
from mindspore import Tensor, nn, ops

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


# main class


class ResidualVQ(nn.Cell):
    """Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        codebook_dim=None,
        shared_codebook=False,
        heads=1,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        accept_image_fmap=False,
        distribute=False,
        **kwargs
    ):
        super().__init__()
        assert heads == 1, "residual vq is not compatible with multi-headed codes"
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Dense(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Dense(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers

        self.accept_image_fmap = accept_image_fmap
        self.layers = nn.CellList(
            [
                VectorQuantize(
                    dim=codebook_dim,
                    codebook_dim=codebook_dim,
                    accept_image_fmap=accept_image_fmap,
                    distribute=distribute,
                    **kwargs
                )
                for _ in range(num_quantizers)
            ]
        )

        assert all([not vq.has_projections for vq in self.layers])

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = (
            quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4
        )

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    @property
    def codebooks(self):
        codebooks = [layer._codebook.embed for layer in self.layers]
        codebooks = ops.stack(codebooks, axis=0)
        codebooks = codebooks.squeeze(1)
        return codebooks

    def get_codes_from_indices(self, indices):
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices_shape = indices.shape
        indices = indices.reshape((indices_shape[0], prod(indices_shape[1:][:-1]), indices_shape[-1]))

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert (
                self.quantize_dropout > 0.0
            ), "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations"
            indices = ops.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        # get ready for gathering

        codebooks = self.codebooks.expand_dims(1).repeat(batch, 1)
        gather_indices = indices.movedim(-1, 0).expand_dims(-1).repeat(codebooks.shape[-1], -1)

        # take care of quantizer dropout

        mask = gather_indices == -1.0
        gather_indices = gather_indices.masked_fill(mask, 0)  # have it fetch a dummy code to be masked out later

        all_codes = codebooks.gather_elements(2, gather_indices)  # gather all codes

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.0)

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        all_codes = all_codes.reshape(all_codes.shape[:2] + indices_shape[1:][:-1] + all_codes.shape[-1:])

        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = codes.sum(0)
        return self.project_out(codes_summed)

    def construct(
        self,
        x,
        mask=None,
        indices=None,
        return_all_codes=False,
        sample_codebook_temp=None,
        freeze_codebook=False,
        rand_quantize_dropout_fixed_seed=None,
    ):
        global ce_losses
        num_quant, quant_dropout_multiple_of, return_loss = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            exists(indices),
        )

        x = self.project_in(x)

        assert not (self.accept_image_fmap and exists(indices))

        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        if return_loss:
            assert not ops.any(
                indices == -1
            ), "some of the residual vq indices were dropped out. please use indices derived when the Cell is in eval mode to derive cross entropy loss"
            ce_losses = []

        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:
            rand = (
                random.Random(rand_quantize_dropout_fixed_seed) if exists(rand_quantize_dropout_fixed_seed) else random
            )

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1
                )

            null_indices_shape = (x.shape[0], *x.shape[-2:]) if self.accept_image_fmap else tuple(x.shape[:2])
            null_indices = ops.full(null_indices_shape, -1.0, dtype=ms.int64)
            null_loss = ops.full((1,), 0.0, dtype=x.dtype)

        # go through the layers

        for quantizer_index, layer in enumerate(self.layers):
            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            quantized, *rest = layer(
                residual,
                mask=mask,
                indices=layer_indices,
                sample_codebook_temp=sample_codebook_temp,
                freeze_codebook=freeze_codebook,
            )

            residual = residual - Tensor(quantized)
            quantized_out = quantized_out + quantized

            if return_loss:
                ce_loss = rest[0]
                ce_losses.append(ce_loss)
                continue

            embed_indices, loss = rest

            all_indices.append(embed_indices)
            all_losses.append(loss)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # whether to early return the cross entropy loss

        if return_loss:
            return quantized_out, sum(ce_losses)

        # stack all losses and indices

        all_losses, all_indices = map(partial(ops.stack, axis=-1), (all_losses, all_indices))

        ret = (quantized_out, all_indices, all_losses)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret


# grouped residual vq


class GroupedResidualVQ(nn.Cell):
    def __init__(self, *, dim, groups=1, accept_image_fmap=False, distribute=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        self.rvqs = nn.CellList([])

        for _ in range(groups):
            self.rvqs.append(
                ResidualVQ(dim=dim_per_group, accept_image_fmap=accept_image_fmap, distribute=distribute, **kwargs)
            )

    @property
    def codebooks(self):
        return ops.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices):
        codes = tuple(rvq.get_codes_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return ops.stack(codes)

    def get_output_from_indices(self, indices):
        outputs = tuple(rvq.get_output_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return ops.cat(outputs, axis=self.split_dim)

    def construct(
        self,
        x,
        indices=None,
        return_all_codes=False,
        sample_codebook_temp=None,
        freeze_codebook=False,
        mask=None,
    ):
        shape, split_dim = x.shape, self.split_dim
        assert shape[split_dim] == self.dim

        # split the feature dimension into groups

        x = x.chunk(self.groups, axis=split_dim)

        indices = default(indices, tuple())
        return_ce_loss = len(indices) > 0
        assert len(indices) == 0 or len(indices) == self.groups

        forward_kwargs = dict(
            return_all_codes=return_all_codes,
            sample_codebook_temp=sample_codebook_temp,
            mask=mask,
            freeze_codebook=freeze_codebook,
            rand_quantize_dropout_fixed_seed=random.randint(0, 1e7),
        )

        # invoke residual vq on each group

        out = tuple(
            rvq(chunk, indices=chunk_indices, **forward_kwargs)
            for rvq, chunk, chunk_indices in zip_longest(self.rvqs, x, indices)
        )
        out = tuple(zip(*out))

        # if returning cross entropy loss to rvq codebooks

        if return_ce_loss:
            quantized, ce_losses = out
            return ops.cat(quantized, axis=split_dim), sum(ce_losses)

        # otherwise, get all the zipped outputs and combine them

        quantized, all_indices, commit_losses, *maybe_all_codes = out

        quantized = ops.cat(quantized, axis=split_dim)
        all_indices = ops.stack(all_indices)
        commit_losses = ops.stack(commit_losses)

        ret = (quantized, all_indices, commit_losses, *maybe_all_codes)
        return ret
