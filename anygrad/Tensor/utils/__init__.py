from .generator import Generator
from .generator import rand
from . import utils_c as C
from ...Tensor.ThHelper import Reshape, float32, float64
from ...tensor import Tensor

from typing import Tuple, Optional
import anygrad

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

def zeros(shape:Tuple[int], requires_grad:Optional[bool] = False, dtype:Optional[anygrad.float32|anygrad.float64] = anygrad.float32) -> Tensor:
    dtype_mapping = {"float32":float32, "float64":float64}
    return __use_ops_zeros_ones(
        shape=shape,
        requires_grad=requires_grad,
        dtype="float32" if dtype == float32 else "float64",
        dtype_mapping=dtype_mapping,
        opration_name="Zeros"
    )

def ones(shape:Tuple[int], requires_grad:Optional[bool] = False, dtype:Optional[anygrad.float32|anygrad.float64] = anygrad.float32) -> Tensor:
    dtype_mapping = {"float32":float32, "float64":float64}
    return __use_ops_zeros_ones(
        shape=shape,
        requires_grad=requires_grad,
        dtype="float32" if dtype == float32 else "float64",
        dtype_mapping=dtype_mapping,
        opration_name="Ones"
    )

def zeros_like(tensor:Tensor, requires_grad:Optional[bool] = None, dtype:Optional[anygrad.float32|anygrad.float64] = anygrad.float32) -> Tensor:
    if dtype is None:
        dtype = float32 if tensor.base.dtype == 'float32' else float64
    requires_grad = requires_grad if requires_grad is not None else tensor.requires_grad
    return zeros(tensor.shape, requires_grad = requires_grad , dtype = dtype)

def ones_like(tensor:Tensor, requires_grad:Optional[bool] = None, dtype:Optional[anygrad.float32|anygrad.float64] = anygrad.float32) -> Tensor:
    if dtype is None:
        dtype = float32 if tensor.base.dtype == 'float32' else float64
    requires_grad = requires_grad if requires_grad is not None else tensor.requires_grad
    return ones(tensor.shape, requires_grad = requires_grad, dtype = dtype)

def log(tensor:Tensor, requires_grad:Optional[bool]=None) -> Tensor:
    return __use_ops_log(
        tensor,
        requires_grad=requires_grad,
        dtype_mapping={"float32":float32, "float64":float64},
        opration_name="Log"
    )
    
def log10(tensor:Tensor, requires_grad:Optional[bool]=None) -> Tensor:
    return __use_ops_log(
        tensor,
        requires_grad=requires_grad,
        dtype_mapping={"float32":float32, "float64":float64},
        opration_name="Log10"
    )

def log2(tensor:Tensor, requires_grad:Optional[bool]=None) -> Tensor:
    return __use_ops_log(
        tensor,
        requires_grad=requires_grad,
        dtype_mapping={"float32":float32, "float64":float64},
        opration_name="Log2"
    )

def exp(tensor:Tensor, requires_grad:Optional[bool]=None) -> Tensor:
    return __use_ops_log(
        tensor,
        requires_grad=requires_grad,
        dtype_mapping={"float32":float32, "float64":float64},
        opration_name="Exp"
    )

def exp2(tensor:Tensor, requires_grad:Optional[bool]=None) -> Tensor:
    return __use_ops_log(
        tensor,
        requires_grad=requires_grad,
        dtype_mapping={"float32":float32, "float64":float64},
        opration_name="Exp2"
    )

__all__ = ["Generator", "rand"]