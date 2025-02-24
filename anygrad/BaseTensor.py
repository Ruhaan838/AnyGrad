from typing import Optional, Tuple, Callable, Any

from .Tensor import tensor_c as C
from .Tensor import ThError as errors
from .Tensor import ThHelper as Th

import anygrad
import anygrad.AutoGrad as Ag

class BaseTensor:
    
    def __init__(self, requires_grad=None):
        
        self.requires_grad = requires_grad and Ag.GradMode.is_enabled()
        self.grad = None
        self.name_backward = ""
        self._backward = lambda : None
        self._prev = set()
        self.is_leaf = True
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.base.shape)

    @property
    def ndim(self) -> int:
        return self.base.ndim

    @property
    def size(self) -> int:
        return self.base.size
            
    @property
    def dtype(self) -> str:
        return self.base.dtype

    def _create_tensor(self, data, shape, TensorClass, requires_grad: bool, sel_dtype):
        reshaped = Th.Reshape()(data, shape)
        if sel_dtype in {Th.int32, Th.int64, Th.bool}:
            return TensorClass(reshaped, dtype=sel_dtype)
        else:
            return TensorClass(reshaped, requires_grad=requires_grad, dtype=sel_dtype)
    @staticmethod    
    def _apply_operation(
        tensor1,
        tensor2:'BaseTensor',
        TensorClass:Any,
        has_scalar: bool,
        operation: Callable,
        operation_name: str,
        broadcast_checker: Callable,
    ):
        dtype_map = {
            "float32":Th.float32,
            "float64":Th.float64,
            "int32":Th.int32,
            "int64":Th.int64,
            "bool":Th.bool
        }
        
        if isinstance(tensor2, (int, float)) and has_scalar:
            data = [operation(i, tensor2) for i in tensor1.base.data]
            dtype = dtype_map[tensor1.base.dtype]
            ans = tensor1._create_tensor(data, tensor1.shape, TensorClass, tensor1.requires_grad, dtype)
            
            if ans.requires_grad:
                ans._prev = {tensor1}
                ans._backward = getattr(Ag.GradientCal, f"{operation_name.capitalize()}_grad")(tensor1, tensor2, ans)
                ans.name_backward = f"<{operation_name}Backward1>"
                ans.is_leaf = False
                
            del data, dtype
            return ans

        allow = broadcast_checker(tensor1.shape, tensor2.shape, tensor1.base.ndim, tensor2.base.ndim)
        errors.broadcast_error(allow, f" {operation_name} we found {tensor1.shape} and {tensor2.shape}")
        
        operation_func = {
            name:getattr(C, f"{operation_name.capitalize()}{name.capitalize()}") for name in dtype_map.keys()
        }
        
        data, shape = operation_func[tensor1.base.dtype](tensor1.base, tensor2.base)
        req = tensor1.requires_grad or tensor2.requires_grad
        ans = tensor1._create_tensor(data, shape, TensorClass, req, dtype_map[tensor1.base.dtype])
        
        if ans.requires_grad:
            ans._prev = {tensor1, tensor2}
            ans.name_backward = f"<{operation_name}Backward0>"
            ans._backward = getattr(Ag.GradientCal, f"{operation_name.capitalize()}_grad")(tensor1, tensor2, ans)
            
            ans.is_leaf = False
        
        del data, shape, req
        return ans
    
    @staticmethod
    def _reduce_ops(
        tensor1,
        TensorClass,
        axis: Optional[int] = -1,
        keepdims: bool = False,
        operation_name: str = "Sum"
    ):
        allow = C.is_sum_allow(axis, tensor1.base.ndim)
        errors.sum_error(allow, f" {operation_name} we found {axis} and {tensor1.base.ndim}")
        
        dtype_map = {
            "float32":Th.float32,
            "float64":Th.float64,
            "int32":Th.int32,
            "int64":Th.int64,
        }
        
        operation_func = {
            name:getattr(C, f"{operation_name.capitalize()}{name.capitalize()}") for name in dtype_map.keys() 
        }
        
        data, shape = operation_func[tensor1.base.dtype](tensor1.base, axis, keepdims)
        ans = tensor1._create_tensor(data, shape, TensorClass, tensor1.requires_grad, dtype_map[tensor1.base.dtype])
        del data, shape
        
        if ans.requires_grad:
            ans._prev = {tensor1}
            ans.name_backward = f"<{operation_name.capitalize()}Backward>"
            ans._backward = getattr(Ag.GradientCal, f"{operation_name}_grad")(tensor1, ans)
            ans.is_leaf = False
            
        return ans

    def __iter__(self):
        return iter(self.data)
    
    def __neg__(self):
        return 0.0 - self
    
    def backward(self, custom_grad=None) -> None:
        if not self.requires_grad:
            raise ValueError("Backward pass only works if requires_grad is True")
        
        if self.shape == (1,):
            if custom_grad is not None:
                raise ValueError("Do not provide a custom gradient for scalar outputs; use a scalar value for grad computation")
            self.grad = anygrad.ones_like(self, requires_grad=False)
        else:
            if custom_grad is None:
                raise ValueError("A custom gradient must be provided for non-scalar outputs")
            if custom_grad.shape != self.shape:
                raise ValueError(f"Custom grad shape {custom_grad.shape} doesn't match output shape {self.shape}")
            self.grad = custom_grad

        topo = Ag.BuildGraph.construct_graph(self)
        
        for v in reversed(topo):
            if v is not self and v._prev:
                v.grad = None
            
        for v in topo:
            v._backward()
