from .tensor import Tensor 
from .Tensor.ThHelper import float32, float64, Reshape
from .AutoGrad import no_grad
from .Tensor import tensor_c as C

def __use_ops_zeros_ones(shape:tuple, requires_grad:bool, dtype:str, dtype_mapping:dict, opration_name:str):
    reshape = Reshape()
    opration_func = {
        "float32":getattr(C, f"{opration_name.capitalize()}Float32"),
        "float64":getattr(C, f"{opration_name.capitalize()}Float64")
    }
    
    data, shape = opration_func[dtype](shape)
    ans = reshape(data, shape)
    del data, shape
    ans = Tensor(ans, dtype = dtype_mapping[dtype], requires_grad=requires_grad)
    return ans

def __use_ops_log(tensor1, requires_grad:bool, dtype_mapping:dict, opration_name:str):
    reshape = Reshape()
    opration_func = {
        "float32":getattr(C, f"{opration_name.capitalize()}Float32"),
        "float64":getattr(C, f"{opration_name.capitalize()}Float64")
    }
    
    data, shape = opration_func[tensor1.base.dtype](tensor1.base)
    ans = reshape(data, shape)
    del data, shape
    ans = Tensor(ans, dtype = dtype_mapping[tensor1.base.dtype], requires_grad=requires_grad)
    return ans

def zeros(shape, requires_grad = False, dtype = float32):
    dtype_mapping = {"float32":float32, "float64":float64}
    return __use_ops_zeros_ones(
        shape=shape,
        requires_grad=requires_grad,
        dtype="float32" if dtype == float32 else "float64",
        dtype_mapping=dtype_mapping,
        opration_name="Zeros"
    )

def ones(shape, requires_grad = False, dtype = float32):
    dtype_mapping = {"float32":float32, "float64":float64}
    return __use_ops_zeros_ones(
        shape=shape,
        requires_grad=requires_grad,
        dtype="float32" if dtype == float32 else "float64",
        dtype_mapping=dtype_mapping,
        opration_name="Ones"
    )

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

def log(tensor, requires_grad=None):
    return __use_ops_log(
        tensor,
        requires_grad=requires_grad,
        dtype_mapping={"float32":float32, "float64":float64},
        opration_name="Log"
    )
    
def log10(tensor, requires_grad=None):
    return __use_ops_log(
        tensor,
        requires_grad=requires_grad,
        dtype_mapping={"float32":float32, "float64":float64},
        opration_name="Log10"
    )

def log2(tensor, requires_grad=None):
    return __use_ops_log(
        tensor,
        requires_grad=requires_grad,
        dtype_mapping={"float32":float32, "float64":float64},
        opration_name="Log2"
    )

def exp(tensor, requires_grad=None):
    return __use_ops_log(
        tensor,
        requires_grad=requires_grad,
        dtype_mapping={"float32":float32, "float64":float64},
        opration_name="Exp"
    )

def exp2(tensor, requires_grad=None):
    return __use_ops_log(
        tensor,
        requires_grad=requires_grad,
        dtype_mapping={"float32":float32, "float64":float64},
        opration_name="Exp2"
    )

__all__ = ["float32", "float64", "no_grad"]