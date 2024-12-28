import anygrad
class GradientCal:
    @staticmethod
    def accumulate_grad(tensor1, tensor2):
        if tensor1.grad.shape != tensor2.shape:
            tensor2.grad = tensor2.sum(axis=0, keepdims=True)
        tensor1.grad += tensor2
    
    @staticmethod
    def Add_grad(tensor1, tensor2, ans_tensor):
        def _backward():
            if tensor1.requires_grad:
                if tensor1.grad is None:
                   tensor1.grad = anygrad.zeros_like(tensor1, requires_grad=False) 
                GradientCal.accumulate_grad(tensor1, ans_tensor.grad)
                
            if not isinstance(tensor2, (int, float)):
                if tensor2.requires_grad:
                    if tensor2.grad is None:
                        tensor2.grad = anygrad.zeros_like(tensor2, requires_grad=False)
                    GradientCal.accumulate_grad(tensor2, ans_tensor.grad)
        return _backward
    
    @staticmethod
    def Sub_grad(tenosr1, tensor2, ans_tensor):
        def _backward():
            if tenosr1.requires_grad:
                if tenosr1.grad is None:
                    tenosr1.grad = anygrad.zeros_like(tenosr1, requires_grad=False)
                GradientCal.accumulate_grad(tenosr1, ans_tensor.grad)
                
            if not isinstance(tensor2, (int, float)):
                if tensor2.requires_grad:
                    if tensor2.grad is None:
                        tensor2.grad = anygrad.zeros_like(tenosr1, requires_grad=False)
                GradientCal.accumulate_grad(tensor2, ans_tensor.grad * -1)
        return _backward

    @staticmethod
    def Mul_grad(tensor1, tensor2, ans_tensor):
        def _backward():
            if tensor1.requires_grad:
                if tensor1.grad is None:
                    tensor1.grad = anygrad.zeros_like(tensor1, requires_grad=False)
                acc = tensor2 * ans_tensor.grad
                acc.requires_grad = False
                GradientCal.accumulate_grad(tensor1, acc)
                
            if not isinstance(tensor2, (int, float)):
                if tensor2.requires_grad:
                    if tensor2.grad is None:
                        tensor2.grad = anygrad.zeros_like(tensor2, requires_grad=False)
                    acc = tensor1 * ans_tensor.grad
                    acc.requires_grad = False
                    GradientCal.accumulate_grad(tensor2, acc)  
            else:
                if tensor1.requires_grad:
                    if tensor1.grad is None:
                        tensor1.grad = anygrad.zeros_like(tensor2, requires_grad=False)
                    acc = tensor2 * ans_tensor.grad
                    acc.requires_grad = False
                    GradientCal.accumulate_grad(tensor1, acc)
                             
        return _backward
    
    @staticmethod
    def Div_grad(tensor1, tensor2, ans_tensor):
        def _backward():
            if tensor1.grad is None:
                tensor1.grad = anygrad.zeros_like(tensor1, requires_grad=False)
            acc = (1 / tensor2) * ans_tensor.grad
            acc.requires_grad = False
            GradientCal.accumulate_grad(tensor1, acc)

            if not isinstance(tensor2, (int, float)):
                if tensor2.requires_grad:
                    if tensor2.grad is None:
                        tensor2.grad = anygrad.zeros_like(tensor2, requires_grad=False)
                    acc = ((-tensor1) / tensor2 ** 2) * ans_tensor.grad
                    acc.requires_grad = False
                    GradientCal.accumulate_grad(tensor2, acc)
                    
        return _backward

    @staticmethod
    def sum_grad(tensor, ans_tensor):
        def _backward():
            
            if tensor.requires_grad:
                if tensor.grad is None:
                    tensor.grad = anygrad.zeros_like(tensor, requires_grad=False)
                tensor.grad += ans_tensor.grad
                
        return _backward
