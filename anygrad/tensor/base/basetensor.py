from typing import Optional, Tuple, Callable, Any

from anygrad.tensor.base import tensor_c as C
from anygrad.tensor.base import ThError as errors
from anygrad.tensor.base import ThHelper as Th

import anygrad
import anygrad.AutoGrad as Ag


class BaseTensor:

    _dtype_map = {
        "float32": Th.float32,
        "float64": Th.float64,
        "int32": Th.int32,
        "int64": Th.int64,
        "bool": Th.bool,
    }

    _prority_dtype = {"bool": 0, "int32": 1, "int64": 2, "float32": 3, "float64": 4}

    @classmethod
    def _sel_dtype(cls, dtype1: str, dtype2: str):
        num1 = cls._prority_dtype[dtype1]
        num2 = cls._prority_dtype[dtype2]
        return dtype1 if num1 > num2 else dtype2

    @classmethod
    def register_dtype(cls, key: str, value: Any):
        cls._dtype_map[key] = value

    def __init__(self, requires_grad=False):

        self.requires_grad = requires_grad and Ag.GradMode.is_enabled()
        self.grad = None
        self.name_backward = ""
        self._backward = lambda: None
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
        return BaseTensor._dtype_map[self.base.dtype]

    @staticmethod
    def _create_tensor(data, shape, TensorClass, requires_grad: bool, sel_dtype):
        reshaped = Th.reshape(data, shape)
        if sel_dtype in {Th.int32, Th.int64, Th.bool}:
            return TensorClass(reshaped, dtype=sel_dtype)
        else:
            return TensorClass(reshaped, requires_grad=requires_grad, dtype=sel_dtype)

    @staticmethod
    def _apply_operation(
        tensor1,
        tensor2,
        TensorClass: Any,
        has_scalar: bool,
        operation: Callable,
        operation_name: str,
        broadcast_checker: Callable,
        OtherClass: Any = None 
    ):
        """
        Apply an element-wise operation on two tensors or a tensor and a scalar.
        
        Parameters:
            tensor1: The first tensor operand.
            tensor2: The second tensor operand or a scalar (if allowed).
            TensorClass: The tensor class to use for the output.
            has_scalar: Whether scalar values are allowed for tensor2.
            operation: A function representing the element-wise operation.
            operation_name: A string denoting the operation (e.g., "Add", "Mul").
            broadcast_checker: A function to verify broadcasting compatibility.
            OtherClass: An alternative output tensor class (optional).

        Returns:
            A new tensor resulting from applying the operation while preserving gradient info.
        """
        if isinstance(tensor2, (int, float)) and has_scalar:
            data = [operation(i, tensor2) for i in tensor1.base.data]
            selected_class = OtherClass if OtherClass is not None else TensorClass
            ans = BaseTensor._create_tensor(
                data, tensor1.shape, selected_class, tensor1.requires_grad, tensor1.dtype
            )
            if ans.requires_grad:
                ans._prev = {tensor1}
                grad_func = getattr(Ag.GradientCal, f"{operation_name.capitalize()}_grad", None)
                if grad_func:
                    ans._backward = grad_func(tensor1, tensor2, ans)
                    ans.name_backward = f"<{operation_name}Backward1>"
                else:
                    ans._backward = lambda: None
                ans.is_leaf = False
            return ans

        allow = broadcast_checker(
            tensor1.shape, tensor2.shape, tensor1.base.ndim, tensor2.base.ndim
        )
        errors.broadcast_error(
            allow, f"{operation_name} operation: incompatible shapes {tensor1.shape} and {tensor2.shape}"
        )

        if tensor1.base.dtype != tensor2.base.dtype:
            sel_dtype = BaseTensor._sel_dtype(tensor1.base.dtype, tensor2.base.dtype)
            if tensor1.base.dtype != sel_dtype:
                tensor1 = anygrad.cast(tensor1, BaseTensor._dtype_map[sel_dtype])
            if tensor2.base.dtype != sel_dtype:
                tensor2 = anygrad.cast(tensor2, BaseTensor._dtype_map[sel_dtype])

        op_func_name = f"{operation_name.capitalize()}{tensor1.base.dtype.capitalize()}"
        try:
            operation_func = getattr(C, op_func_name)
        except AttributeError as e:
            raise NotImplementedError(f"Operation {op_func_name} not implemented in C++") from e

        data, shape = operation_func(tensor1.base, tensor2.base)
        requires_grad = tensor1.requires_grad or tensor2.requires_grad
        unified_dtype = BaseTensor._dtype_map[
            BaseTensor._sel_dtype(tensor1.base.dtype, tensor2.base.dtype)
        ]
        if OtherClass is not None:
            unified_dtype = anygrad.float32 if unified_dtype == anygrad.int32 else anygrad.float64
            ans = BaseTensor._create_tensor(
                data, shape, OtherClass, requires_grad=requires_grad, sel_dtype=unified_dtype
            )
        else:
            ans = BaseTensor._create_tensor(
                data, shape, TensorClass, requires_grad=requires_grad, sel_dtype=unified_dtype
            )

        if ans.requires_grad:
            ans._prev = {tensor1, tensor2}
            grad_func = getattr(Ag.GradientCal, f"{operation_name.capitalize()}_grad", None)
            if grad_func:
                ans._backward = grad_func(tensor1, tensor2, ans)
                ans.name_backward = f"<{operation_name}Backward0>"
            else:
                ans._backward = lambda: None
            ans.is_leaf = False

        return ans

    @staticmethod
    def _reduce_ops(
        tensor1, TensorClass, axis: Optional[int], keepdims: bool, operation_name: str, OtherClass:Any=None
    ):
        allow = C.is_sum_allow(axis, tensor1.base.ndim)
        errors.sum_error(
            allow, f" {operation_name} we found {axis} and {tensor1.base.ndim}"
        )

        try:
            operation_func = getattr(
                C, f"{operation_name.capitalize()}{tensor1.base.dtype.capitalize()}"
            )
        except Exception:
            pass

        data, shape = operation_func(tensor1.base, axis, keepdims)
        
        if OtherClass is not None:
            ans = BaseTensor._create_tensor(
                data, shape, OtherClass, tensor1.requires_grad, tensor1.shape
            )
        else:
            ans = BaseTensor._create_tensor(
                data, shape, TensorClass, tensor1.requires_grad, tensor1.dtype
            )

        if ans.requires_grad:
            ans._prev = {tensor1}
            ans._backward = getattr(
                Ag.GradientCal, f"{operation_name.capitalize()}_grad"
            )(tensor1, ans)
            ans.name_backward = f"<{operation_name}Backward0>"
            ans.is_leaf = False

        del data, shape
        return ans

    @staticmethod
    def _trans_ops(tensor1, dim0: int, dim1: int, TensorClass: Any):

        if dim0 < 0 and dim1 < 0:
            dim0 = tensor1.ndim + dim0
            dim1 = tensor1.ndim + dim1

        allow = False if tensor1.ndim < 2 else True
        errors.dim_error(allow, f" Transpose we found {tensor1.ndim}")
        try:
            opration_func = getattr(C, f"Trans{tensor1.base.dtype.capitalize()}")
        except Exception:
            pass
        data, shape = opration_func(tensor1.base, dim0, dim1)
        ans = BaseTensor._create_tensor(
            data, shape, TensorClass, tensor1.requires_grad, tensor1.dtype
        )

        if ans.requires_grad:
            ans._prev = {tensor1}
            ans.name_backward = "<TransBackward0>"
            ans._backward = getattr(Ag.GradientCal, "Trans_grad")(tensor1, ans)
            ans.is_leaf = False

        return ans
    
    @staticmethod
    def _apply_view(tensor, shape, TensorClass):
        allow = C.is_view_allow(tensor.base.shape, tensor.base.size)
        errors.dim_error(allow, f" View we found {shape} invoke with {tensor.base.size}")
        
        try:
            operation_func = getattr(C, f"View{tensor.base.dtype.capitalize()}")
        except Exception:
            pass
        
        data, shape = operation_func(tensor.base, shape)
        ans = BaseTensor._create_tensor(
            data=data, shape=shape, TensorClass=TensorClass, requires_grad=tensor.requires_grad, sel_dtype=tensor.dtype
        )
        
        if ans.requires_grad:
            ans._prev = {tensor}
            ans.name_backward = "<ViewBackward0>"
            ans._backward = getattr(Ag.GradientCal, "View_grad")(tensor, ans)
            ans.is_leaf = False
            
        return ans
        
    def __iter__(self):
        return iter(self.data)

    def __neg__(self):
        return -1 * self

    def backward(self, custom_grad=None) -> None:
        if not self.requires_grad:
            raise ValueError("Backward pass only works if requires_grad is True")

        if self.shape == (1,):
            if custom_grad is not None:
                raise ValueError(
                    "Do not provide a custom gradient for scalar outputs; use a scalar value for grad computation"
                )
            self.grad = anygrad.ones_like(self, requires_grad=False, dtype=self.dtype)
        else:
            if custom_grad is None:
                raise ValueError(
                    "A custom gradient must be provided for non-scalar outputs"
                )
            if custom_grad.shape != self.shape:
                raise ValueError(
                    f"Custom grad shape {custom_grad.shape} doesn't match output shape {self.shape}"
                )
            self.grad = custom_grad

        topo = Ag.BuildGraph.construct_graph(self)

        for v in reversed(topo):
            if v is not self and v._prev:
                v.grad = None

        for v in topo:
            v._backward()
