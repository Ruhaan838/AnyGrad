import os
import time
import sys

class Tester:
    def __init__(self, test_list, actual_list):
        self.test_list = test_list
        self.actual_list = actual_list
        assert len(self.actual_list) == len(self.test_list), "The test_list and actual_list have same size"
        self.num_tests = len(test_list)
        
    def display(self, current_index=None, result=None):

        os.system('cls' if os.name == 'nt' else 'clear')
        try:
            with os.popen('stty size', 'r') as console:
                _, display_width = map(int, console.read().split())
            display_width = max(20, display_width - 100)  
        except:
            display_width = 50  
        print()
        print("-" * display_width)

        for i, func_name in enumerate(self.test_list):
            if current_index is not None and i == current_index:
                status = f"-> Running '{func_name}'"
            elif result is not None and i < current_index:
                status = f"✅'{func_name}' completed: {result[i]}"
            else:
                status = f"  '{func_name}' pending"
            print(status)
        
        print("-" * display_width)

    def eval_fn(self, func_name, *args, **kwargs):

        func = globals().get(func_name)
        if func and callable(func):
            return func(*args, **kwargs)
        else:
            raise ValueError(f"'{func_name}' is not a valid function.")
    
    def test(self):
        results = []
        for current_index, func_name in enumerate(self.test_list):

            self.display(current_index=current_index, result=results)
            time.sleep(0.5) 
            try:
                from_module = self.eval_fn(func_name)
                actual = self.eval_fn(self.actual_list[current_index])
                
                if from_module == actual:
                    results.append(f"✅ Pass test {func_name}")
                else:
                    results.append(f"❌ Failed test {func_name}")
                
            except Exception as e:
                raise ValueError(f"Error: {e}")
            
            self.display(current_index=current_index + 1, result=results)
            time.sleep(1)  
            
        print("All tests completed. ✅".center(10))