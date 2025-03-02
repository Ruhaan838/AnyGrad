from typing import Optional, Tuple, Callable, Any

from . import tensor_c as C
from . import ThError as errors
from . import ThHelper as Th

import anygrad
import anygrad.AutoGrad as Ag

class BaseTensor:

    _DTYPE_MAP: dict[str, Any] = {
        "float32": Th.float32,
        "float64": Th.float64,
        "int32": Th.int32,
        "int64": Th.int64,
        "bool": Th.bool
    }
    _OPERATION_REGISTRY: dict[str, Callable] = {}  

    @classmethod
    def register_dtype(cls, key: str, value: Any):
        cls._DTYPE_MAP[key] = value

    @classmethod
    def register_operation(cls, op_name: str, func: Callable):
        cls._OPERATION_REGISTRY[op_name.lower()] = func

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
        return BaseTensor._DTYPE_MAP[self.base.dtype]
    
    @staticmethod
    def _create_tensor(data, shape, TensorClass, requires_grad: bool, sel_dtype):
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
        
        if isinstance(tensor2, (int, float)) and has_scalar:
            data = [operation(i, tensor2) for i in tensor1.base.data]
            ans = BaseTensor._create_tensor(data, tensor1.shape, TensorClass, tensor1.requires_grad, tensor1.dtype)
            
            if ans.requires_grad:
                ans._prev = {tensor1}
                ans._backward = getattr(Ag.GradientCal, f"{operation_name.capitalize()}_grad")(tensor1, tensor2, ans)
                ans.name_backward = f"<{operation_name}Backward1>"
                ans.is_leaf = False
                
            del data
            return ans

        if not isinstance(tensor2, (int, float)):
            if tensor1.dtype != tensor2.dtype:
                if tensor1.dtype == Th.float64 or tensor2.dtype == Th.float64:
                    target_dtype = Th.float64
                    if tensor1.dtype != Th.float64:
                        tensor1 = TensorClass(tensor1.data, requires_grad=tensor1.requires_grad, dtype=Th.float64)
                    if tensor2.dtype != Th.float64:
                        tensor2 = TensorClass(tensor2.data, requires_grad=tensor2.requires_grad, dtype=Th.float64)

        allow = broadcast_checker(tensor1.shape, tensor2.shape, tensor1.base.ndim, tensor2.base.ndim)
        errors.broadcast_error(allow, f" {operation_name} we found {tensor1.shape} and {tensor2.shape}")
        
        op_key = operation_name.lower()
        if op_key in BaseTensor._OPERATION_REGISTRY:
            op_func = BaseTensor._OPERATION_REGISTRY[op_key]
        else:
            op_func = None
        
        operation_func = {}
        for name in BaseTensor._DTYPE_MAP.keys():
            try:
                if op_func:
                    operation_func[name] = op_func
                else:
                    operation_func[name] = getattr(C, f"{operation_name.capitalize()}{name.capitalize()}")
            except Exception:
                pass
        
        data, shape = operation_func[tensor1.base.dtype](tensor1.base, tensor2.base)
        req = tensor1.requires_grad or tensor2.requires_grad
        reshaped = Th.Reshape()(data, shape)
        
        ans = BaseTensor._create_tensor(data, shape, TensorClass, req, tensor1.dtype)
        
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
        
        operation_func = {}
        for name in ("float32", "float64", "int32", "int64"):
            try:
                operation_func[name] = getattr(C, f"{operation_name.capitalize()}{name.capitalize()}")
            except Exception as e:
                pass
        
        data, shape = operation_func[tensor1.base.dtype](tensor1.base, axis, keepdims)
        reshaped = Th.Reshape()(data, shape)
        
        if tensor1.base.dtype in ("int32", "int64", "bool"):
            ans = TensorClass(reshaped, dtype=tensor1.base.dtype)
        else:
            ans = TensorClass(reshaped, requires_grad=tensor1.requires_grad, dtype=tensor1.base.dtype)
        
        if ans.requires_grad:
            ans._prev = {tensor1}
            ans.name_backward = f"<{operation_name.capitalize()}Backward>"
            ans._backward = getattr(Ag.GradientCal, f"{operation_name}_grad")(tensor1, ans)
            ans.is_leaf = False
            
        del data, shape
        return ans
    
    @staticmethod
    def _trans_ops(tensor1, dim0:int, dim1:int, TensorClass:Any):

        if dim0 < 0 and dim1 < 0:
            dim0 = tensor1.ndim + dim0
            dim1 = tensor1.ndim + dim1
        
        allow = False if tensor1.ndim < 2 else True
        errors.dim_error(allow, f" Transpose we found {tensor1.ndim}")
        try:
            opration_func = getattr(C, f"Trans{tensor1.base.dtype.capitalize()}")
        except:
            pass
        data, shape = opration_func(tensor1.base, dim0, dim1)
        ans = BaseTensor._create_tensor(data, shape, TensorClass, tensor1.requires_grad, tensor1.dtype)
        
        if ans.requires_grad:
            ans._prev = {tensor1}
            ans.name_backward = "<TransBackward0>"
            ans._backward = getattr(Ag.GradientCal, "Trans_grad")(tensor1, ans)
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
            self.grad = anygrad.ones_like(self, requires_grad=False, dtype=self.dtype)
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
