from typing import NewType
from .tensor_c import float32, float64
from collections.abc import Iterable, Sequence

float32 = NewType('float32', float32)
float64 = NewType('float64', float64)

class TensorConvert:
    def __call__(self, data):
            if isinstance(data, list):
                return [self.__call__(ele) for ele in data]
            else:
                return float(data)

class TensorType:
    def __init__(self, dtype):
        self.dtype = str(dtype) if isinstance(dtype, str) else dtype

    def __call__(self, data):
        valid_types = {float32, float64}  
        if self.dtype not in valid_types:
            raise TypeError("Tensor must have a valid dtype: 'float32', 'float64', float32, or float64.")
class ToList:
    def __call__(self, data):
        def flatten(lst):
            for item in lst:
                if isinstance(item, list):
                    yield from flatten(item)
                else:
                    yield item
        
        if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            return list(flatten(data))
        else:
            return data

class CalShape:
    def __call__(self, data, shape=()):
        if not isinstance(data, Sequence):
            return shape
        
        if isinstance(data[0], Sequence):
            l = len(data[0])
            if not all(len(item) == l for item in data):
                raise ValueError("Not all list have the same Length of you data")
        
        shape += (len(data), )
        shape = self.__call__(data[0], shape)
        
        return shape

class Reshape:
    def __call__(self, data, shape):
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        if len(data) != total_elements:
            raise ValueError(f"List lenght '{len(data)}' is not mathch with new shape '{shape}'")
        
        def create_reshape(data, current_shape):
            if len(current_shape) == 1:
                return data[:current_shape[0]]
            
            sublist = []
            chunck_size = len(data) // current_shape[0]
            
            for i in range(current_shape[0]):
                start_idx = i * chunck_size
                end_idx = start_idx + chunck_size
                sublist.append(create_reshape(data[start_idx:end_idx], current_shape[1:]))
            return sublist
        return create_reshape(data, shape)

def round_list(data):
    def process(item):
        if isinstance(item, list):
            return [process(sub) for sub in item]
        return round(item, 3)
    return process(data)
