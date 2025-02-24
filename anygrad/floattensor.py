from typing import Optional, List
import pprint

from .Tensor import tensor_c as C
from .Tensor import ThHelper as Th
from .BaseTensor import BaseTensor

import anygrad.AutoGrad as Ag

class FloatTensor(BaseTensor):
    """
        Class to repesent a FloatTensor.

        Attributes
        ----------
        data : List | Tuple
            Any Iterable 
        
        requires_grad: Optional[bool] = False
            if `True` the gradient caculation is happend
            
        dtype: Optional[anygrad.float32 | anygrad.float64] = anygrad.float32

        Methods
        ----------
        data:
            return the item of the tensor in list form.
        shape:
            return the shape of the tensor in tuple form.
        ndim:
            return the dim of the tensor in int form.
        requires_grad:
            return the bool value if the requires_grad.
        grad:
            a tensor value that allow you to see the gredient of the tensor.

        add(other):
            other: Tensor | int | float
            add the Tensor or number.

        sub(other):
            other: Tensor | int | float
            sub the Tensor or number.

        mul(other):
            other: Tensor | int | float
            mul the Tensor or number.

        div(other):
            other: Tensor | int | float
            div the Tensor or number.

        pow(other):
            other: Tensor | int | float
            pow the Tensor or number.
            
        matmul(other):
            other: Tensor
            matrix multiplication of the Two valid shape tensor.
        
        sum(self, axis: Optional[int] = -1, keepdims: Optional[bool] = False) -> Tensor:
            sum the tensor with axis and keepdims.
            
        backward(self, custom_grad:Optional[Tensor] = None) -> None:
            Do the backward pass if the requires-grad is true for given tensor.
        
    """
    def __init__(self, data: List[int], requires_grad=True, dtype:Optional[Th.float32 |Th.float64] = Th.float32):
        super().__init__(requires_grad)
        
        list_data = Th.ToList()(data)
        
        if dtype in {Th.float32, Th.float64}:
            convert_data = Th.TensorConvertFloat()
        else:
            raise ValueError("Invalid dtype provided")
        
        self.data = convert_data(data)
        
        shape = Th.CalShape()(data)
        
        if dtype == Th.float32:
            self.base = C.float32(self.data, shape)   
        elif dtype == Th.float64:
            self.base = C.float64(self.data, shape)
    
    def __repr__(self):
        data = Th.round_list(self.data)
        formate_data = pprint.pformat(data, width=25, depth=35)
        base_str = f"Tensor({formate_data}"
        
        if self.name_backward:
            return base_str + f" name_backward = {self.name_backward})"
        
        if self.requires_grad:
            return base_str + f" requires_grad = {self.requires_grad})"
        
        if self.base.dtype != "float32":
            return base_str + f" dtype= {self.dtype})"
        return base_str + ")"
        
    def __add__(self, other) -> 'FloatTensor':
        return BaseTensor._apply_operation(
            self,
            other,
            FloatTensor,
            True,
            operation=lambda x, y: x + y,
            operation_name="Add",
            broadcast_checker=C.isbroadcast,
        )
    
    def __radd__(self, other) -> 'FloatTensor':
        return self._add__(other)
    
    def __sub__(self, other) -> 'FloatTensor':
        return BaseTensor._apply_operation(
            self,
            other,
            FloatTensor,
            True,
            operation=lambda x, y: x - y,
            operation_name="Sub",
            broadcast_checker=C.isbroadcast,
        )
        
    def __rsub__(self, other) -> 'FloatTensor':
        return self.__sub__(other)
    
    def __mul__(self, other) -> 'FloatTensor':
        return BaseTensor._apply_operation(
            self,
            other,
            FloatTensor,
            True,
            operation=lambda x, y: x * y,
            operation_name="Mul",
            broadcast_checker=C.isbroadcast,
        )
        
    def __rmul__(self, other) -> 'FloatTensor':
        return self.__mul__(other)
    
    def __truediv__(self, other) -> 'FloatTensor':
        return BaseTensor._apply_operation(
            self,
            other,
            FloatTensor,
            True,
            operation=lambda x, y: x / y,
            operation_name="Div",
            broadcast_checker=C.isbroadcast,
        )
        
    def __rtruediv__(self, other) -> 'FloatTensor':
        return self.__truediv__(other)
    
    def __pow__(self, other) -> 'FloatTensor':
        return BaseTensor._apply_operation(
            self,
            other,
            FloatTensor,
            True,
            operation=lambda x, y: x ** y,
            operation_name="Pow",
            broadcast_checker=C.isbroadcast,
        )
        
    def __matmul__(self, other) -> 'FloatTensor':
        return BaseTensor._apply_operation(
            self,
            other,
            FloatTensor,
            False,
            operation=lambda :None,
            operation_name="Matmul",
            broadcast_checker=C.is_matmul_broadcast,
        )
    
    def sum(self, axis:Optional[int] = -1, keepdims: Optional[bool] = False):
        return BaseTensor._reduce_ops(self, FloatTensor, axis, keepdims, "Sum")
    
    def mean(self, axis:Optional[int] = -1, keepdims: Optional[bool] = False):
        return BaseTensor._reduce_ops(self, FloatTensor, axis, keepdims, "Mean")
    
    def min(self, axis:Optional[int] = -1, keepdims: Optional[bool] = False):
        return BaseTensor._reduce_ops(self, FloatTensor, axis, keepdims, "Min")
    
    def max(self, axis:Optional[int] = -1, keepdims: Optional[bool] = False):
        return BaseTensor._reduce_ops(self, FloatTensor, axis, keepdims, "Max")
    
    def median(self, axis:Optional[int] = -1, keepdims: Optional[bool] = False):
        return BaseTensor._reduce_ops(self, FloatTensor, axis, keepdims, "Median")
    