"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from math import prod
from typing import Any, List, Optional, Union

import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, Tensor, ops
from mindspore.nn import Cell

# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


# tensor helpers


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + Tensor(zhat - z)


# main class


class FSQ(Cell):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self._levels = Tensor(levels, dtype=ms.int32)

        self._basis = ops.cumprod(Tensor([1] + levels[:-1]), dim=0, dtype=ms.int32)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(self._levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Dense(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Dense(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        self.implicit_codebook = Parameter(
            self.indices_to_codes(ops.arange(self.codebook_size), project_out=False), requires_grad=False
        )

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = ops.where(self._levels % 2 == 0, Tensor(0.5), Tensor(0.0))
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width).to(ms.float32) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(axis=-1).to(ms.int32)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = indices.expand_dims(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes_shape = codes.shape
            codes = codes.reshape(codes_shape[:-2] + (codes_shape[-1] * codes_shape[-2],))

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = codes.movedim(-1, 1)

        return codes

    def construct(self, z: Tensor) -> tuple[Any, Union[Tensor, Any]]:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        global z_shape0
        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = z.movedim(1, -1)
            z_shape0 = z.shape
            z = z.reshape((z_shape0[0], int(prod(z_shape0[1:][:-1])), z_shape0[-1]))

        assert z.shape[-1] == self.dim, f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z_shape = z.shape
        z = z.reshape(z_shape[:2] + (self.num_codebooks, int(z_shape[-1] / self.num_codebooks)))

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes_shape = codes.shape
        codes = codes.reshape(codes_shape[:-2] + (codes_shape[-1] * codes_shape[-2],))

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = out.reshape((out.shape[0],) + z_shape0[1:][:-1] + (out.shape[-1],))
            out = out.movedim(-1, 1)

            indices = indices.reshape((indices.shape[0],) + z_shape0[1:][:-1] + (indices.shape[-1],))

        if not self.keep_num_codebooks_dim:
            indices = indices.squeeze(-1)

        return out, indices
