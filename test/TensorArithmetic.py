import anygrad


a = anygrad.Tensor([1,2,3], requires_grad=True)
b = anygrad.Tensor([1,2,3], requires_grad=True)
c = a + b