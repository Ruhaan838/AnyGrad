import subprocess

# path = os.getcwd()
# subprocess.call(["gcc", "-c", f"{path}/anygrad/clib/ThAllocate.c", "-o", "ThAllocate.o"], cwd=f"{path}/anygrad/clib")

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import setup, find_packages
except Exception as e:
    print(f"Not find the pybind11 so installing due to {e}")
    subprocess.call(["pip", "install", "pybind11"])
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import setup, find_packages
    
__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "anygrad.Tensor.tensor_c",
        [
            "anygrad/Tensor/bind_tensor.cpp",
            "anygrad/Tensor/clib/ThAllocate.cpp",
            "anygrad/Tensor/clib/ThBaseops.cpp",
            "anygrad/Tensor/clib/Thhelpers.cpp",
            "anygrad/Tensor/utils/anygrad_utils.cpp"
        ],
        language="c++",
        extra_compile_args=["-O2", "-std=c++20"]
    ),
    # Pybind11Extension(
    #     "anygrad.AutoGrad.autograd_c",
    #     [
    #         "anygrad/AutoGrad/bind_autograd.cpp",
    #         "anygrad/AutoGrad/clib/grad_helper.cpp",
    #     ],
    #     language="c++",
    #     extra_compile_args=["-O2", "-std=c++20"]
    # )
]

setup(
    name="anygrad",
    version=__version__,
    author="Ruhaan",
    ext_modules=ext_modules,
    cmdclass={"build_ext":build_ext},
    zip_safe=False,
    packages=find_packages(),
    package_dir={"": "."}, 
    package_data={
        "anygrad": ["Tensor/*.py", "__init__.py", "anygrad/*.py"], 
    } 
)