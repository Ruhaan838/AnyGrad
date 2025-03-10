import anygrad
import numpy as np

from Test_template import TestTemplate

data = anygrad.randint(2,3, low=2, high=10).data

float_dtypes = {
    "float32":(anygrad.float32, np.float32),
    "float64":(anygrad.float64, np.float64),
}
int_dtype = {
    "int32":(anygrad.int32, np.int32),
    "int64":(anygrad.int64, np.int64)
}
dtypes = {
    "float32":(anygrad.float32, np.float32),
    "float64":(anygrad.float64, np.float64),
    "int32":(anygrad.int32, np.int32),
    "int64":(anygrad.int64, np.int64),
}


shaped_ops_float_ops = ["rand", "ones", "zeros"]
shaped_ops_int_ops = ["randint", "ones", "zeros"]
tensor_ops = ["log10", "log2", "exp", "exp2", "ones_like", "zeros_like"]

def shaped_float_func(dtype):
    
    result = {}
    shape = (3,4,5)
    
    for op in shaped_ops_float_ops:
        
        ans = getattr(anygrad, f"{op}")(*shape, dtype=dtypes[dtype][0])
        out1 = ans.shape
        
        result[op] = (out1, shape)
        
    return result

def shaped_int_func(dtype):
    
    result = {}
    shape = (3,4,5)
    
    for op in shaped_ops_int_ops:
        if op == "randint":
            ans = getattr(anygrad, f"{op}")(*shape, dtype=dtypes[dtype][0], high=10)
        else:
            ans = getattr(anygrad, f"{op}")(*shape, dtype=dtypes[dtype][0])
        out1 = ans.shape
        
        result[op] = (out1, shape)
        
    return result

def tensor_func(dtype):
    
    result = {}
    tensor1 = anygrad.Tensor(data, dtype=dtypes[dtype][0])
    arr1 = np.array(data, dtype=dtypes[dtype][1])
    for op in tensor_ops:
        if op in {"ones_like", "zeros_like"}:
            ans = getattr(anygrad, f"{op}")(tensor1, dtype=tensor1.dtype)
        else:
            ans = getattr(anygrad, f"{op}")(tensor1)
            
        out1 = np.round(ans.data, 2).tolist()
        
        ans = getattr(np, f"{op}")(arr1)
        out2 = np.round(ans, 2).tolist()
        
        result[op] = (out1, out2)
        
    return result

tester = TestTemplate("Initialization", dtype_map=float_dtypes)
tester.test_console(shaped_float_func, "Initialization mismatch")

tester = TestTemplate("Initialization", dtype_map=int_dtype)
tester.test_console(shaped_int_func, "Initialization mismatch")

tester = TestTemplate("Tensor Operations", dtypes)
tester.test_console(tensor_func, "Tensor Operations")