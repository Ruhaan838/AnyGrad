from .tensor import Tensor
from .floattensor import FloatTensor
from .inttensor import IntTensor
from .booltensor import BoolTensor
from .Tensor.ThHelper import float32, float64, int32, int64, bool
from .AutoGrad import no_grad
from .Tensor.utils import Generator, rand, ones, ones_like, zeros, zeros_like, log, log10, log2, exp, exp2
from .version import __version__

def matmul(tensor1, tensor2):
    return tensor1 @ tensor2



__all__ = ["Tensor", "FloatTensor", "IntTensor", "BoolTensor",
           "float32", "float64", "int32", "int64", "bool",
           "no_grad", "Generator", "rand", "ones", "ones_like", "zeros", "zeros_like",
           "log", "log10", "log2", "exp", "exp2", "matmul", "__version__"]