from typing import NewType
from . import tensor_c as C
from collections.abc import Iterable, Sequence

float32 = NewType('float32', C.float32)
float32.__module__ = "anygrad"
float64 = NewType('float64', C.float64)
float64.__module__ = "anygrad"
int32 = NewType('int32', C.int32)
int32.__module__ = "anygrad"
int64 = NewType('int64', C.int64)
int64.__module__ = "anygrad"
bool = NewType('bool', C.bool)
bool.__module__ = "anygrad"

class TensorConvertInt:
    def __call__(self, data):
        if isinstance(data, list):
            return [self.__call__(ele) for ele in data]
        else:
            return int(data)
class TensorConvertFloat:
    def __call__(self, data):
            if isinstance(data, list):
                return [self.__call__(ele) for ele in data]
            else:
                return float(data)

class TensorConvertBool:
    def __call__(self, data):
        if isinstance(data, list):
            return [self.__call__(ele) for ele in data]
        else:
            return bool(data)
        
class ValidDataType:
    def __init__(self):
        self._list = {
            float: 0,
            int: 0,
            bool: 0
        }

    def __call__(self, data):
        if not isinstance(data, list):
            if type(data) in self._list:
                self._list[type(data)] += 1
            else:
                self._list[type(data)] = 1
        else:
            for ele in data:
                self.__call__(ele)
        return max(self._list, key=lambda x:str(x))
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

def round_list(data, round_factor=3):
    def process(item):
        if isinstance(item, list):
            return [process(sub) for sub in item]
        return round(item, round_factor)
    return process(data)
