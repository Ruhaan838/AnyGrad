from .tensor.tensor import Tensor
from .tensor.floattensor import FloatTensor
from .tensor.inttensor import IntTensor
from .tensor.booltensor import BoolTensor
from .tensor.ThHelper import (float32, float64, 
                              int32, int64, 
                              bool)
from .AutoGrad import no_grad
from .utils import (Generator, rand, randint, 
                    ones, ones_like, 
                    zeros, zeros_like, 
                    log, log10, log2, exp, exp2)
from .version import __version__

def matmul(tensor1, tensor2):
    return tensor1 @ tensor2

def cast(tensor:Tensor, target_dtype):
    return Tensor(tensor.data, requires_grad=tensor.requires_grad, dtype=target_dtype)

__all__ = ["Tensor", "FloatTensor", "IntTensor", "BoolTensor",
           "float32", "float64", "int32", "int64", "bool",
           "no_grad", "Generator", "rand", "randint", "ones", "ones_like", "zeros", "zeros_like",
           "log", "log10", "log2", "exp", "exp2", "matmul", "cast", "__version__"]