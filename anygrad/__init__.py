from .tensor import Tensor 
from .Tensor.ThHelper import float32, float64
from .AutoGrad import no_grad
from .Tensor.utils import Generator, rand, ones, ones_like, zeros, zeros_like, log, log10, log2, exp, exp2

def matmul(tensor1, tensor2):
    return tensor1 @ tensor2

__all__ = ["float32", "float64", "no_grad","Tensor","Generator", "rand", "ones", "ones_like", "zeros","zeros_like", "log", "log10", "log2","exp", "exp2"
]