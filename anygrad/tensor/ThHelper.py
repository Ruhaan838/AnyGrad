from typing import NewType
from . import tensor_c as C
from collections.abc import Iterable, Sequence

float32 = NewType("float32", C.float32)
float32.__module__ = "anygrad"
float64 = NewType("float64", C.float64)
float64.__module__ = "anygrad"
int32 = NewType("int32", C.int32)
int32.__module__ = "anygrad"
int64 = NewType("int64", C.int64)
int64.__module__ = "anygrad"
bool = NewType("bool", C.bool)
bool.__module__ = "anygrad"

# use the chatGPT to make the these functions fast


def convert_tensor(data, conv_type):
    if type(data) is not list:
        return conv_type(data)
    return [convert_tensor(ele, conv_type) for ele in data]


def valid_data_type(data):
    counts = {float: 0, int: 0, bool: 0}
    stack = [data]

    while stack:
        current = stack.pop()
        if type(current) is list:
            stack.extend(current)
        else:
            t = type(current)
            counts[t] = counts.get(t, 0) + 1

    return max(counts, key=lambda x: str(x))


def flat_list(data):
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        result = []
        stack = [data]
        while stack:
            current = stack.pop()
            if isinstance(current, list):
                stack.extend(reversed(current))
            else:
                result.append(current)
        return result
    return data


def cal_shape(data):
    shape = []
    while isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if not data:
            shape.append(0)
            break
        if isinstance(data[0], Sequence):
            expected_len = len(data[0])
            if any(len(item) != expected_len for item in data):
                raise ValueError("Not all lists have the same length in your data")
        shape.append(len(data))
        data = data[0]
    return tuple(shape)


def reshape(data, shape):
    total_elements = 1
    for dim in shape:
        total_elements *= dim

    if len(data) != total_elements:
        raise ValueError(
            f"List length '{len(data)}' does not match new shape '{shape}'"
        )

    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    return _reshape_flat(data, shape, strides, 0)


def _reshape_flat(data, shape, strides, offset):
    if len(shape) == 1:
        return data[offset: offset + shape[0]]
    sublist = []
    cur_stride = strides[0]
    for i in range(shape[0]):
        sublist.append(
            _reshape_flat(data, shape[1:], strides[1:], offset + i * cur_stride)
        )
    return sublist


def round_list(data, round_factor=4):
    if isinstance(data, list):
        return [round_list(item, round_factor) for item in data]
    return round(data, round_factor)


