import anygrad

class GradMode:
    _enable = True
    @classmethod
    def is_enabled(cls):
        return cls._enable
    
    @classmethod
    def set_enable(cls, mode:bool):
        cls._enable = mode

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
                   tensor1.grad = anygrad.zeros(tensor1.shape) 
                GradientCal.accumulate_grad(tensor1, ans_tensor)
                
            if tensor2.requires_grad:
                if tensor2.grad is None:
                    tensor2.grad = anygrad.zeros(tensor2.shape)
                GradientCal.accumulate_grad(tensor2, ans_tensor)
                
        return _backward    

    @staticmethod
    def sum_grad(tensor, ans_tensor):
        def _backward():
            
            if tensor.requires_grad:
                if tensor.grad is None:
                    tensor.grad = anygrad.zeros(tensor.shape)
                tensor.grad += ans_tensor.grad
                
        return _backward

class BuildGrad:
    @staticmethod
    def construct_graph(tensor):
        tensor.grad = anygrad.ones(tensor.shape)
        
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(tensor)
        return topo