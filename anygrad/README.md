## anygrad

### [autograd](anygrad\autograd)
The current version of autograd is completely in `Python` including graph construction but a later version of `anygrad` change and uses the other engine's backward pass or the `C++`.
```text
    > clib
    > __init__.py
    > autograd.py
    > bind_autograd.cpp
```

### [tensor](anygrad\tensor)
The tensor folder contains so many things like the `Tensor`, `FloatTensor`, `IntTensor`, and `BoolTensor`.  
This is how the one Tensor class is made.

So all tensors have one common class `BaseTensor`. This `BaseTensor` is the base class of `FloatTensor`, `IntTensor`, and `BoolTensor`.
And these three classes made the one main class `Tensor`.

