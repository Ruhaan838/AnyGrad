from .tensor import Tensor 
from .Tensor.ThHelper import float32, float64, Reshape
from .AutoGrad import no_grad
from .Tensor import tensor_c as C

def zeros(shape, requires_grad = False, dtype = float32):
    result, shape = C.zerosfloat32(shape) if dtype == float32 else C.zerosfloat64(shape)
    reshape = Reshape()
    result = reshape(result, shape)
    ans = Tensor(result, dtype=dtype, requires_grad=requires_grad)
    return ans

def ones(shape, requires_grad = False, dtype = float32):
    result, shape = C.onesfloat32(shape) if dtype == float32 else C.onesfloat64(shape)
    reshape = Reshape()
    result = reshape(result, shape)
    ans = Tensor(result, dtype=dtype, requires_grad=requires_grad)
    return ans

def zeros_like(tensor, dtype = None, requires_grad = None):
    if dtype is None:
        dtype = float32 if tensor.base.dtype == 'float32' else float64
    requires_grad = requires_grad if requires_grad is not None else tensor.requires_grad
    return zeros(tensor.shape, requires_grad = requires_grad , dtype = dtype)

def ones_like(tensor, dtype=None, requires_grad = None):
    if dtype is None:
        dtype = float32 if tensor.base.dtype == 'float32' else float64
    requires_grad = requires_grad if requires_grad is not None else tensor.requires_grad
    return ones(tensor.shape, requires_grad = requires_grad, dtype = dtype)

__all__ = ["Tensor", "float32", "float64"]