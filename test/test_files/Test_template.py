from typing import Any, AnyStr, Dict, Callable
import anygrad
import numpy as np
import os
import time
import subprocess
import sys

class WrongAnswerError(Exception):
    __module__ = ""



class TestTemplate:
    def __init__(self, test_name: AnyStr, dtype_map: Dict = None, tolerance: float = 1e-5) -> None:
        self.test_name = test_name
        self.tolerance = tolerance
        self.dtype_map = {
            "float32": (anygrad.float32, np.float32),
            "float64": (anygrad.float64, np.float64),
            "int32": (anygrad.int32, np.int32),
            "int64": (anygrad.int64, np.int64),
        } if dtype_map is None else dtype_map

    def _arrays_equal(self, arr1, arr2) -> bool:
        return np.allclose(np.array(arr1), np.array(arr2), rtol=self.tolerance, atol=self.tolerance)

    def _format_output(self, passed: bool, test_name: str, actual, expected) -> str:
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        if passed:
            return f"{GREEN} {test_name} = Accepted{RESET}"
        return f"{RED} {test_name} = Fail\nActual: {actual}\nExpected: {expected}{RESET}"

    def test_console(self, ops_func: Callable, error_msg: AnyStr):
        print("\n" + "="*20 + f" Test - {self.test_name} " + "="*20)
        
        for test, dtype in enumerate(self.dtype_map.keys(), 1):
            print(f"\n Test {test}: {dtype}")
            
            results = ops_func(dtype)
            
            for op_name, (ag_result, np_result) in results.items():
                passed = self._arrays_equal(ag_result, np_result)
                print(self._format_output(passed, op_name, ag_result, np_result))
                if not passed:
                    raise WrongAnswerError(f"{error_msg}\nOperation: {op_name}")
        
    def _write_file(self, file_name: AnyStr, message: AnyStr = ""):
        
        if os.path.exists(file_name):
            with open(file_name, "a") as f:
                f.write(message)
        else:        
            with open(file_name, "w") as f:
                f.write(message)
    
    def test_file(self, ops_func: Callable, error_msg: AnyStr):
        filename = f"test_results_{self.test_name}.txt"
        
        with open(filename, "w") as f:
            f.write(f"Test Results - {self.test_name}\n{'='*50}\n")
            
            for test, dtype in enumerate(self.dtype_map.keys(), 1):
                f.write(f"\nTest Group {test}: {dtype}\n{'-'*30}\n")
                
                results = ops_func(dtype)
                
                for op_name, (ag_result, np_result) in results.items():
                    passed = self._arrays_equal(ag_result, np_result)
                    result = "PASS " if passed else f"FAIL\nActual: {ag_result}\nExpected: {np_result}"
                    f.write(f"{op_name}: {result}\n")
                    if not passed:
                        raise WrongAnswerError(f"{error_msg}\nOperation: {op_name}")
        
        print(f"Results saved to {filename}")
