import anygrad
import numpy as np
from Test_template import TestTemplate

data = anygrad.randint(2,3,low=1,high=10).data
other = anygrad.randint(2,3,low=1,high=3).data

dtypes = {
    "float32":(anygrad.float32, np.float32),
    "float64":(anygrad.float64, np.float64),
    "int32":(anygrad.int32, np.int32),
    "int64":(anygrad.int64, np.int64),
}

arithmetic_ops = ["add", "sub", "mul", "truediv", "pow"]
reduction_ops = ["sum", "mean", "min", "max", "median"]

def arithmetic_func(dtype):

    results = {}
    
    for op in arithmetic_ops:
        tensor1 = anygrad.Tensor(data, dtype=dtypes[dtype][0])
        tensor2 = anygrad.Tensor(other, dtype=dtypes[dtype][0])
        ans = getattr(tensor1, f"__{op}__")(tensor2)
        out1 = np.round(ans.data, 2).tolist()
        
        arr1 = np.array(data, dtype=dtypes[dtype][1])
        arr2 = np.array(other, dtype=dtypes[dtype][1])
        ans = getattr(arr1, f"__{op}__")(arr2)
        out2 = np.round(ans, 2).tolist()
        
        results[op] = (out1, out2)
    
    return results

def arithetic_scaler_func(dtype):
    
    result = {}
    
    for op in arithmetic_ops:
        tensor1 = anygrad.Tensor(data, dtype=dtypes[dtype][0])
        ans = getattr(tensor1, f"__{op}__")(5)
        out1 = np.round(ans.data, 2).tolist()
        
        arr1 = np.array(data, dtype=dtypes[dtype][1])
        ans = getattr(arr1, f"__{op}__")(5)
        out2 = np.round(ans, 2).tolist()
        
        result[op] = (out1, out2)
        
    return result

def reduction_func(dtype):

    results = {}
    
    for op in reduction_ops:
        tensor1 = anygrad.Tensor(data, dtype=dtypes[dtype][0])
        ans = getattr(tensor1, f"{op}")()
        out1 = np.round(ans.data, 2).tolist()
        if isinstance(out1, list) and len(out1) == 1:
            out1 = out1[0]
        
        arr1 = np.array(data, dtype=dtypes[dtype][1])
        ans = getattr(np, f"{op}")(arr1)
        out2 = float(np.round(ans, 2))
        
        results[op] = (out1, out2)
    
    return results

tester = TestTemplate("Arithmetic", dtype_map=dtypes)
tester.test_console(arithmetic_func, "Arithmetic operation mismatch")

tester = TestTemplate("Arithmetic Scaler", dtype_map=dtypes)
tester.test_console(arithetic_scaler_func, "Arithmetic Reduction operation mismatch")

tester = TestTemplate("Reduction", dtype_map=dtypes)
tester.test_console(reduction_func, "Reduction operation mismatch")