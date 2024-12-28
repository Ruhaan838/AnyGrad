from typing import Union, List, Optional
import pprint
import warnings

from .Tensor import tensor_c as C
from .Tensor import ThHelper as Th
from .Tensor import ThError as errors

import anygrad
import anygrad.AutoGrad as Ag

class Tensor():
    def __init__(self, 
                 data: Union[List[float], List[float]], 
                 requires_grad:Optional[bool] = False, 
                 dtype: Optional[Th.float32 | Th.float64] = Th.float32):
        
        #Convert the Nd list to 1D list for Backend
        list_data = Th.ToList()
        list_data = list_data(data)
        
        #check the Type for the data
        check = Th.TensorType(dtype)
        check(data)
        
        #convert the data to float
        convert_data = Th.TensorConvert()
        self.data = convert_data(data)
        
        #caculate the shape of the Data for backend
        shape = Th.CalShape()
        shape = shape(data)

        #call the backend to init the Tensor
        if dtype == Th.float32:
            self.base = C.float32(list_data, shape)
        elif dtype == Th.float64:
            self.base = C.float64(list_data, shape)
        else:
            raise TypeError("Unsupported dtype. Use float32 or float64.")
        
        self.requires_grad = requires_grad and Ag.GradMode.is_enabled()
        self.grad = None
        self.name_backward = ""
        self._backward = lambda : None
        self._prev = set()
        self.is_leaf = True
        
    def __repr__(self):
        formate_data = pprint.pformat(self.data, width=30, underscore_numbers=True, depth=40)
        base_str = f"Tensor({formate_data}"
        
        if self.name_backward:
            return base_str + f" name_backward = {self.name_backward})"
        
        if self.requires_grad:
            return base_str + f" requires_grad = {self.requires_grad})"
        
        if self.base.dtype != "float32":
            return base_str + f" dtype= {self.dtype})"
        return base_str + ")"
    
    def __add__(self, other):
        reshape = Th.Reshape()
        
        if isinstance(other, (int, float)):
            data = [i + other for i in self.base.data]
            ans = reshape(data, self.shape)
            del(data)
            dtype = Th.float32 if self.base.dtype == 'float32' else Th.float64
            ans = Tensor(ans, dtype=dtype, requires_grad=self.requires_grad)
            if ans.requires_grad:
                ans._prev = {self}
                ans._backward = Ag.GradientCal.add_grad(self, other, ans)
                ans.is_leaf = False
                
            return ans
        
        allow = C.isbroadcast(self.shape, other.shape, self.base.ndim, other.base.ndim)
        errors.broadcast_error(allow, f" add(+) we found {self.base.shape} and {other.base.shape}")

        if self.base.dtype == "float32":
            data, shape = C.AddFloat32(self.base, other.base)
            ans = reshape(data, shape)
            del(data); del(shape)
            req = self.requires_grad or other.requires_grad
            ans = Tensor(ans, dtype=Th.float32, requires_grad=req)
            
            if ans.requires_grad:
                ans._prev = {self, other}
                ans.name_backward = "<AddBackward0>"
                ans._backward = Ag.GradientCal.add_grad(self, other, ans)
                ans.is_leaf = False
            
            return ans
        
        elif self.base.dtype == "float64":
            data, shape = C.AddFloat64(self.base, other.base)
            ans = reshape(data, shape)
            del(data); del(shape)
            req = self.requires_grad or other.requires_grad
            ans = Tensor(ans, dtype=Th.float64, requires_grad=req)
            
            if ans.requires_grad:
                ans._prev = {self, other}
                ans.name_backward = "<AddBackward0>"
                ans._backward = Ag.GradientCal.add_grad(self, other, ans)
                ans.is_leaf = False
                
            return ans
        
    def __hash__(self): return id(Tensor)
    
    @property
    def shape(self): return tuple(self.base.shape)

    @property
    def ndim(self): return self.base.ndim

    @property
    def size(self): return self.base.size
            
    @property
    def dtype(self): return self.base.dtype
    
    def sum(self, axis: Optional[int] = -1, keepdims: Optional[bool] = False):
        reshape = Th.Reshape()
        allow = C.is_sum_allow(axis, self.base.ndim)
        errors.sum_error(allow, f" sum() we found {axis} and {self.base.ndim}")
        
        if self.base.dtype == "float32":
            
            data, shape = C.SumFloat32(self.base, axis, keepdims)
            ans = reshape(data, shape)
            del(data); del(shape)  
            ans = Tensor(ans, dtype=Th.float32, requires_grad=self.requires_grad)
            
            if ans.requires_grad:
                ans._prev = {self}
                ans.name_backward = "<SumBackward0>"
                ans._backward = Ag.GradientCal.sum_grad(self, ans)
                ans.is_leaf = False
            
            return ans
        
        elif self.base.dtype == "float64":
            data, shape = C.SumFloat64(self.base, axis, keepdims)
            ans = reshape(data, shape)
            del(data); del(shape)
            ans = Tensor(ans, dtype=Th.float64, requires_grad=self.requires_grad)
            
            if ans.requires_grad:
                ans._prev = {self}
                ans.name_backward = "<SumBackward1>"
                ans._backward = Ag.GradientCal.sum_grad(self, ans)
                ans.is_leaf = False
                
            return ans

    def backward(self, custom_grad=None):
        
        if self.requires_grad == False:
            raise ValueError("The Backward pass is work only if the requires_grad is True")
        
        if self.shape == (1,):
            if custom_grad is not None:
                raise ValueError("Custom gradient should not be provided for scalar outputs or use the scaler value to compute the grad")
            self.grad = anygrad.ones_like(self, requires_grad=False)
        else:
            if custom_grad is None:
                raise ValueError("Custom gradient must be provided for non-scalar outputs")
            if custom_grad.shape != self.shape:
                raise ValueError(f"Custom grad shape {custom_grad.shape} doesn't match output shape {self.shape}")
            self.grad = custom_grad
        
        topo = Ag.BuildGraph.construct_graph(self)
        
        for v in reversed(topo):
            if v is not self and v._prev:
                # if v.grad is not None:
                #     warnings.warn(f"Gradient should be None for non-leaf node: {v}")
                v.grad = None
        
        for v in topo:
            v._backward()