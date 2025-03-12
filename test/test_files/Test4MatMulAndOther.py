import anygrad
import numpy as np
from Test_template import TestTemplate

data = anygrad.randint(20, 10, high=50).data
other = anygrad.randint(10, 10, high=25).data

dtypes = {
    "float32":(anygrad.float32, np.float32),
    "float64":(anygrad.float64, np.float64),
    "int32":(anygrad.int32, np.int32),
    "int64":(anygrad.int64, np.int64),
}

def matmul_func(dtype):
    
    result = {}
    
    tensor1 = anygrad.Tensor(data, dtype=dtypes[dtype][0])
    tensor2 = anygrad.Tensor(other, dtype=dtypes[dtype][0])
    
    ans = getattr(tensor1, "__matmul__")(tensor2)
    out1 = np.round(ans.data, 2).tolist()
    
    arr1 = np.array(data, dtype=dtypes[dtype][1])
    arr2 = np.array(other, dtype=dtypes[dtype][1])
    ans = getattr(arr1, "__matmul__")(arr2)
    out2 = np.round(ans, 2).tolist()
    
    result["matmul"] = (out1, out2)
    
    return result

def arange_func(dtype):
    
    result = {}
    
    tensor1 = anygrad.Tensor(data, dtype=dtypes[dtype][0])
    ans = getattr(tensor1, "transpose")(0, 1)
    out1 = np.round(ans.data, 2).tolist()
    
    arr1 = np.array(data, dtype=dtypes[dtype][1])
    ans = getattr(arr1, "T")
    out2 = np.round(ans, 2).tolist()
    
    result["transpose"] = (out1, out2)
    
    ans = getattr(tensor1, "view")(10, 20)
    out1 = ans.shape
    out2 = (10, 20)
    result["view"] = (out1, out2)
    
    return result
    
       
tester = TestTemplate("Matrix Multiplication", dtype_map=dtypes)
tester.test_console(matmul_func, "Matrix Multiplication mismatch")

tester = TestTemplate("Arrange", dtype_map=dtypes)
tester.test_console(arange_func, "Arrange mismatch")