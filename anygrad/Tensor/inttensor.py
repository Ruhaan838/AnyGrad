from typing import Optional, List, Union
import pprint

from . import tensor_c as C
from . import ThHelper as Th
from .basetensor import BaseTensor

class IntTensor(BaseTensor):
    """
        Class to repesent a IntTensor.

        Attributes
        ----------
        data : List | Tuple
            Any Iterable 
            
        dtype: Optional[anygrad.int32 | anygrad.int64] = anygrad.int32

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
    def __init__(self, data: List[int], dtype:Optional[Th.int32 | Th.int64] = Th.int32):
        super().__init__()
        
        if isinstance(data, (int, float)):
            data = [data]
        list_data = Th.ToList()(data)
        if isinstance(dtype, str): 
            dtype = getattr(Th, dtype)
        convert_data = Th.TensorConvertInt()
        self.data = convert_data(data)
        shape = Th.CalShape()(data)
        
        if dtype == Th.int32:
            self.base = C.int32(list_data, shape)   
        elif dtype == Th.int64:
            self.base = C.int64(list_data, shape)
    
    def __repr__(self) -> str:
        data = Th.round_list(self.data)
        format_data = pprint.pformat(data, width=25, depth=35)
        base_str = f"Tensor({format_data}"
        return base_str + f", dtype={self.dtype})"
    
    def __getitem__(self, index:Union[int, slice]) -> 'IntTensor':
        new_data = self.data[index]
        return IntTensor(new_data, dtype=self.dtype)
        
    def __add__(self, other) -> 'IntTensor':
        return BaseTensor._apply_operation(
            self,
            other,
            IntTensor,
            True,
            operation=lambda x, y: x + y,
            operation_name="Add",
            broadcast_checker=C.isbroadcast
        )
    
    def __radd__(self, other) -> 'IntTensor':
        return self._add__(other)
    
    def __sub__(self, other) -> 'IntTensor':
        return BaseTensor._apply_operation(
            self,
            other,
            IntTensor,
            True,
            operation=lambda x, y: x - y,
            operation_name="Sub",
            broadcast_checker=C.isbroadcast
        )
        
    def __rsub__(self, other) -> 'IntTensor':
        return self.__sub__(other)
    
    def __mul__(self, other) -> 'IntTensor':
        return BaseTensor._apply_operation(
            self,
            other,
            IntTensor,
            True,
            operation=lambda x, y: x * y,
            operation_name="Mul",
            broadcast_checker=C.isbroadcast
        )
        
    def __rmul__(self, other) -> 'IntTensor':
        return self.__mul__(other)
    
    def __truediv__(self, other) -> 'IntTensor':
        return BaseTensor._apply_operation(
            self,
            other,
            IntTensor,
            True,
            operation=lambda x, y: x / y,
            operation_name="Div",
            broadcast_checker=C.isbroadcast
        )
        
    def __rtruediv__(self, other) -> 'IntTensor':
        return self.__truediv__(other)
    
    def __pow__(self, other) -> 'IntTensor':
        return BaseTensor._apply_operation(
            other,
            IntTensor,
            True,
            operation=lambda x, y: x ** y,
            operation_name="Pow",
            broadcast_checker=C.isbroadcast
        )
        
    def __matmul__(self, other) -> 'IntTensor':
        return BaseTensor._apply_operation(
            self,
            other,
            IntTensor,
            False,
            operation=lambda :None,
            operation_name="Matmul",
            broadcast_checker=C.is_matmul_broadcast
        )
    
    def sum(self, axis:Optional[int] = -1, keepdims: Optional[bool] = False):
        return BaseTensor._reduce_ops(self, IntTensor, axis, keepdims, "Sum")
    
    def mean(self, axis:Optional[int] = -1, keepdims: Optional[bool] = False):
        return BaseTensor._reduce_ops(self, IntTensor, axis, keepdims, "Mean")
    
    def min(self, axis:Optional[int] = -1, keepdims: Optional[bool] = False):
        return BaseTensor._reduce_ops(self, IntTensor, axis, keepdims, "Min")
    
    def max(self, axis:Optional[int] = -1, keepdims: Optional[bool] = False):
        return BaseTensor._reduce_ops(self, IntTensor, axis, keepdims, "Max")
    
    def median(self, axis:Optional[int] = -1, keepdims: Optional[bool] = False):
        return BaseTensor._reduce_ops(self, IntTensor, axis, keepdims, "Median")
    
    def transpose(self, dim0:int, dim1:int) -> 'IntTensor':
        return BaseTensor._trans_ops(self, dim0, dim1, IntTensor)
    
    __module__ = "anygrad"
