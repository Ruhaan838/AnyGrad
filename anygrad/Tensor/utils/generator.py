from . import utils_c as C_
from ...Tensor.ThHelper import Reshape, float32, float64
from ...tensor import Tensor
import random

class Generator(C_.GeneratorBase):
    def __init__(self, seed):
        super().__init__(seed)
    def manual_seed(self, seed):
        return super().manual_seed(seed)
    
def _use_utils_ops(shape, requires_grad, generator, opration_name, dtype):
    reshape = Reshape()
    dtype_mapping = {"float32":float32, "float64":float64}
    opration_func = {
        "float32":getattr(C_, f"{opration_name}Float32"),
        "float64":getattr(C_, f"{opration_name}Float64")
    }
    
    data, shape = opration_func[dtype](shape, generator)
    ans = reshape(data, shape)
    del data, shape
    ans = Tensor(ans, requires_grad, dtype=dtype_mapping[dtype])
    return ans

def rand(shape, generator=None, requires_grad=False, dtype=float32):
    if generator is None:
        generator = C_.GeneratorBase(random.randint(0, 100))
    return _use_utils_ops(
        shape,
        generator=generator,
        requires_grad=requires_grad,
        opration_name="rand",
        dtype="float32" if dtype == float32 else "float64",
    )
