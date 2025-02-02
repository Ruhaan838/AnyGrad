<div align="center">

# AnyGrad

</div>

## Overview
The AnyGrad is a Tensor Library that allow you to do the forward and backward pass very easy.
It's use the C++ as backend and the Python for frontend and do the tensor opration in C++. To combine the both things I use the pybind11.
## Installation
clone the repo:
```bash
git clone https://github.com/Ruhaan838/AnyGrad.git
```
```bash
cd AnyGrad
pip install .
```
## Getting Started
### Create Tensor
```python
import anygrad
a = anygrad.Tensor([1,2,3]) # not calculate the grad
b = anygrad.Tensor([2,3,4], requires_grad=True) # now it's caculate the grad
c = anygrad.Tensor([2,3,4], dtype=anygrad.float64) #create the float64 tensor
```
### Arithmetic
#### Element wise Operations
```python
# do the ele wise opration
d = a + b
d = a * d
d = d / 10
e = e - 10
```
#### Matrix multiplication
```python
a = anygrad.ones((1,2,3), requires_grad=True)
b = anygrad.ones((2,3,4), requires_grad=True)
c = a @ b # shape (2, 2, 4)
c = anygrad.matmul(a, b) #other way
```
### Gradient caculation
```python
a = anygrad.Tensor([1,2,3], requires_grad=True)
b = anygrad.Tensor([2,3,4], requires_grad=True)
c = a * b 
c.sum().backward() #need a sclaer value for backward
print(a.grad) #anygrad.Tensor
print(b.grad) #anygrad.Tensor
```

## Contributing

- you cacn contribute for building docs or to improve the performnce.

## License
see the [LICENSE](LICENSE) file 