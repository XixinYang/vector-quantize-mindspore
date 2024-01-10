from math import prod

from vector_quantize.vector_quantize import VectorQuantize

from mindspore import Tensor, nn, ops
from mindspore.common.initializer import XavierNormal, initializer

from .utils import einsum_ms


def exists(val):
    return val is not None


class RandomProjectionQuantizer(nn.Cell):
    """https://arxiv.org/abs/2202.01855"""

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        codebook_dim,
        num_codebooks=1,
        norm=True,
        distribute=False,
        **kwargs
    ):
        super().__init__()
        self.num_codebooks = num_codebooks

        self.rand_projs = initializer(XavierNormal(), (num_codebooks, dim, codebook_dim))

        # in section 3 of https://arxiv.org/abs/2202.01855
        # "The input data is normalized to have 0 mean and standard deviation of 1 ... to prevent collapse"

        self.norm = norm
        self.norm_func = (
            ops.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5) if norm else nn.Identity()
        )

        self.vq = VectorQuantize(
            dim=codebook_dim * num_codebooks,
            heads=num_codebooks,
            codebook_size=codebook_size,
            use_cosine_sim=True,
            separate_codebook_per_head=True,
            distribute=distribute,
            **kwargs,
        )

    def construct(self, x, indices=None):
        return_loss = exists(indices)

        x = (
            self.norm_func(x, Tensor([1.0] * x.shape[-1]), Tensor([0.0] * x.shape[-1]))[0]
            if self.norm
            else self.norm_func(x)
        )

        x = einsum_ms("b n d, h d e -> b n h e", x, self.rand_projs)
        x_shape = x.shape
        x = x.reshape(x_shape[:2] + (prod(x_shape[2:]),))

        self.vq.set_train(False)
        out = self.vq(x, indices=indices)

        if return_loss:
            _, ce_loss = out
            return ce_loss

        _, indices, _ = out
        return indices
