import numpy as np
from mindspore import Tensor


def einsum_ms(operation, var1, var2):
    var1 = var1.asnumpy() if isinstance(var1, Tensor) else var1
    var2 = var2.asnumpy() if isinstance(var2, Tensor) else var2
    return Tensor(np.einsum(operation, var1, var2))
