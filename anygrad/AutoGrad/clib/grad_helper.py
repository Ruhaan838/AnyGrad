import anygrad
class GradientCal:
    @staticmethod
    def accumulate_grad(tensor1, tensor2):
        if tensor1.grad.shape != tensor2.grad.shape:
            tensor2.grad = tensor2.grad.sum(axis=0, keepdims=True)
        tensor1.grad += tensor2.grad
    
    @staticmethod
    def add_grad(tensor1, tensor2, ans_tensor):
        def _backward():
            if tensor1.requires_grad:
                if tensor1.grad is None:
                   tensor1.grad = anygrad.zeros_like(tensor1, requires_grad=False) 
                GradientCal.accumulate_grad(tensor1, ans_tensor)
                
            if tensor2.requires_grad:
                if tensor2.grad is None:
                    tensor2.grad = anygrad.zeros_like(tensor2, requires_grad=False)
                GradientCal.accumulate_grad(tensor2, ans_tensor)
        return _backward


    @staticmethod
    def sum_grad(tensor, ans_tensor):
        def _backward():
            
            if tensor.requires_grad:
                if tensor.grad is None:
                    tensor.grad = anygrad.zeros_like(tensor, requires_grad=False)
                tensor.grad += ans_tensor.grad
                
        return _backward
