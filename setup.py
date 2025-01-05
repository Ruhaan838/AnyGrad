import subprocess

import sys

compile_args = ["-O2", "-std=c++20"] if sys.platform != "win32" else ["/O2", "/std:c++20"]
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
            "anygrad/Tensor/clib/Thgemm.cpp"
            
        ],
        language="c++",
        extra_compile_args=compile_args
    ),
    Pybind11Extension(
        "anygrad.Tensor.utils.utils_c",
        [
            "anygrad/Tensor/utils/utils_bind.cpp",
            "anygrad/Tensor/utils/random_num.cpp",
            "anygrad/Tensor/clib/Thhelpers.cpp",
            "anygrad/Tensor/utils/init_ops.cpp",
            "anygrad/Tensor/utils/log_arithmetic.cpp",
        ],
        language="c++",
        extra_compile_args=compile_args
    )
]

setup(
    name="anygrad",
    version=__version__,
    description="A module that allow user to do the Tensor opration.",
    long_description=open('README.md', encoding='utf-8').read(),
    author="Ruhaan",
    author_email="ruhaan123dalal@gmail.com",
    license="Apache License",
    ext_modules=ext_modules,
    cmdclass={"build_ext":build_ext},
    zip_safe=False,
    packages=find_packages(),
    package_dir={"": "."}, 
    package_data={
        "anygrad":["Tensor/*.py", "__init__.py", 
                   "anygrad/*.py", "Tensor/utils/*.py"],
    },
    include_package_data=True
)