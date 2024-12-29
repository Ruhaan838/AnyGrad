import anygrad

print("-"*10, " Float32 ", "-"*10)
print("-"*5, " Add ", "-"*5)
a = anygrad.Tensor([1,2,3])
b = anygrad.Tensor([1,2,3])
c = a + b
print("a + b = ", c)
print()
print("-"*5, " Sub ", "-"*5)
a = anygrad.Tensor([1,2,3])
b = anygrad.Tensor([1,2,3])
c = a - b
print("a - b = ", c)
print()
print("-"*5, " Mul ", "-"*5)
a = anygrad.Tensor([1,2,3])
b = anygrad.Tensor([1,2,3])
c = a * b
print("a * b = ", c)
print()
print("-"*5, " Div ", "-"*5)
a = anygrad.Tensor([1,2,3])
b = anygrad.Tensor([1,2,3])
c = a / b
print("a / b = ", c)
print()
print("-"*5, " Pow ", "-"*5)
a = anygrad.Tensor([1,2,3])
b = anygrad.Tensor([1,2,3])
c = a ** b
print("a ** b = ", c)

print("-"*20)

print("-"*10, " Float64 ", "-"*10)
print("-"*5, " Add ", "-"*5)
a = anygrad.Tensor([1,2,3], dtype=anygrad.float64)
b = anygrad.Tensor([1,2,3], dtype=anygrad.float64)
c = a + b
print("a + b = ", c)
print()
print("-"*5, " Sub ", "-"*5)
a = anygrad.Tensor([1,2,3], dtype=anygrad.float64)
b = anygrad.Tensor([1,2,3], dtype=anygrad.float64)
c = a - b
print("a - b = ", c)
print()
print("-"*5, " Mul ", "-"*5)
a = anygrad.Tensor([1,2,3], dtype=anygrad.float64)
b = anygrad.Tensor([1,2,3], dtype=anygrad.float64)
c = a * b
print("a * b = ", c)
print()
print("-"*5, " Div ", "-"*5)
a = anygrad.Tensor([1,2,3], dtype=anygrad.float64)
b = anygrad.Tensor([1,2,3], dtype=anygrad.float64)
c = a / b
print("a / b = ", c)
print()
print("-"*5, " Pow ", "-"*5)
a = anygrad.Tensor([1,2,3], dtype=anygrad.float64)
b = anygrad.Tensor([1,2,3], dtype=anygrad.float64)
c = a ** b
print("a ** b = ", c)