import anygrad
import numpy as np
from Test_template import TestTemplate

data = anygrad.rand(2,3).data
other = anygrad.rand(2,3).data

dtypes = {
    "float32":(anygrad.float32),
    "float64":(anygrad.float64),
}

arithmetic_ops = ["add", "sub", "mul", "truediv", "pow"]
reduction_ops = ["sum", "mean", "min", "max", "median"]

def arithmetic_backward(dtype):
    
    results = {}
    
    for op in arithmetic_ops:
        tensor1 = anygrad.Tensor(data, requires_grad=True, dtype=dtypes[dtype])
        tensor2 = anygrad.Tensor(other, requires_grad=True, dtype=dtypes[dtype])
        ans = getattr(tensor1, f"__{op}__")(tensor2)
        ans.sum().backward()
        out1 = np.round(tensor1.grad.data, 2).tolist()
        
        if op == "add":
            out2 = anygrad.ones_like(tensor1)
        elif op == "sub":
            out2 = anygrad.ones_like(tensor1)
        elif op == "mul":
            out2 = tensor2
        elif op == "truediv":
            out2 = 1.0 / tensor2
        elif op == "pow":
            out2 = tensor2 * tensor1.__pow__(tensor2 - 1)
        
        results[op] = (out1, np.round(out2.data, 2).tolist())
        ans.grad.zero_()
    
    return results

def arithmetic_scaler_backward(dtype):
    
    results = {}
    
    for op in arithmetic_ops:
        tensor1 = anygrad.Tensor(data, requires_grad=True, dtype=dtypes[dtype])
        tensor2 = 3
        ans = getattr(tensor1, f"__{op}__")(tensor2)
        ans.sum().backward()
        out1 = np.round(tensor1.grad.data, 2).tolist()
        
        if op == "add":
            out2 = anygrad.ones_like(tensor1)
        elif op == "sub":
            out2 = anygrad.ones_like(tensor1)
        elif op == "mul":
            out2 = anygrad.ones_like(tensor1) * tensor2
        elif op == "truediv":
            out2 = 1.0 / (anygrad.ones_like(tensor1) * tensor2)
        elif op == "pow":
            out2 = tensor2 * tensor1.__pow__(tensor2 - 1)
        
        results[op] = (out1, np.round(out2.data, 2).tolist())
        ans.grad.zero_()
    
    return results

tester = TestTemplate("Arithmetic Backward", dtype_map=dtypes)
tester.test_console(arithmetic_backward, "Arithmetic backward operation mismatch")

tester = TestTemplate("Arithmetic Scaler Backward", dtype_map=dtypes)
tester.test_console(arithmetic_scaler_backward, "Arithmetic Scaler backward operation mismatch")

