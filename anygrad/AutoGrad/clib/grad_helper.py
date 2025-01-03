import anygrad
class GradientCal:
    @staticmethod
    def initialize_grad(tensor):
        """Ensure tensor.grad is initialized."""
        if tensor.grad is None:
            tensor.grad = anygrad.zeros_like(tensor, requires_grad=False)
    
    @staticmethod
    def accumulate_grad_with_factor(tensor, factor):
        """Accumulate gradients with a scaling factor."""
        if tensor.requires_grad:
            GradientCal.initialize_grad(tensor)
            GradientCal.accumulate_grad(tensor, factor)
            
    @staticmethod
    def accumulate_grad_for_matmul(tensor, factor):
        if tensor.requires_grad:
            GradientCal.initialize_grad(tensor)
            GradientCal.accmulate_grad_matmul(tensor, factor)
        
    @staticmethod
    def accmulate_grad_matmul(tensor1, tensor2):
        """Accumulate gradients for tensor1 using tensor2 for matmul."""
        tensor2.requires_grad = False
        current = tensor2
        
        while current.ndim > tensor1.grad.ndim:
            current = current.sum(0)
        tensor1.grad += current

    @staticmethod
    def accumulate_grad(tensor1, tensor2):
        """Accumulate gradients for tensor1 using tensor2."""
        tensor2.requires_grad = False
        if tensor1.grad.shape != tensor2.shape:
            tensor2.grad = tensor2.sum(axis=0, keepdims=True)
        tensor1.grad += tensor2

    @staticmethod
    def Add_grad(tensor1, tensor2, ans_tensor):
        def _backward():
            GradientCal.accumulate_grad_with_factor(tensor1, ans_tensor.grad)
            if not isinstance(tensor2, (int, float)):
                GradientCal.accumulate_grad_with_factor(tensor2, ans_tensor.grad)
        return _backward

    @staticmethod
    def Sub_grad(tensor1, tensor2, ans_tensor):
        def _backward():
            GradientCal.accumulate_grad_with_factor(tensor1, ans_tensor.grad)
            if not isinstance(tensor2, (int, float)):
                GradientCal.accumulate_grad_with_factor(tensor2, -ans_tensor.grad)
        return _backward

    @staticmethod
    def Mul_grad(tensor1, tensor2, ans_tensor):
        def _backward():
            if tensor1.requires_grad:
                GradientCal.accumulate_grad_with_factor(tensor1, tensor2 * ans_tensor.grad)
            if not isinstance(tensor2, (int, float)) and tensor2.requires_grad:
                GradientCal.accumulate_grad_with_factor(tensor2, tensor1 * ans_tensor.grad)
        return _backward

    @staticmethod
    def Div_grad(tensor1, tensor2, ans_tensor):
        def _backward():
            if tensor1.requires_grad:
                GradientCal.accumulate_grad_with_factor(tensor1, (1 / tensor2) * ans_tensor.grad)
            if not isinstance(tensor2, (int, float)) and tensor2.requires_grad:
                GradientCal.accumulate_grad_with_factor(tensor2, ((-tensor1) / tensor2**2) * ans_tensor.grad)
        return _backward

    @staticmethod
    def Pow_grad(tensor1, tensor2, ans_tensor):
        def _backward():
            if tensor1.requires_grad:
                GradientCal.accumulate_grad_with_factor(tensor1, tensor2 * ans_tensor.grad * (ans_tensor / tensor1))
            if not isinstance(tensor2, (int, float)) and tensor2.requires_grad:
                GradientCal.accumulate_grad_with_factor(tensor2, ans_tensor.grad * ans_tensor * anygrad.log(tensor1))
        return _backward
    
    @staticmethod
    def Matmul_grad(tensor1, tensor2, ans_tensor):
        def _backward():
            if tensor1.requires_grad:
                GradientCal.accumulate_grad_for_matmul(tensor1, ans_tensor.grad @ tensor2.transpose(-2, -1))
            if tensor2.requires_grad:
                GradientCal.accumulate_grad_for_matmul(tensor2, tensor1.transpose(-2, -1) @ ans_tensor.grad)
        return _backward
    
    @staticmethod
    def Trans_grad(tensor1, ans_tensor):
        def _backward():
            if tensor1.requires_grad:
                GradientCal.accumulate_grad_with_factor(tensor1, ans_tensor.grad.transpose(-2, -1))
        return _backward

    @staticmethod
    def sum_grad(tensor, ans_tensor):
        def _backward():
            if tensor.requires_grad:
                GradientCal.accumulate_grad_with_factor(tensor, ans_tensor.grad)
        return _backward
