from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

compile_args = ["-O2", "-std=c++20"] if sys.platform != "win32" else ["/O2", "/std:c++20"]
version = "0.0.2"

ext_modules = [
    Pybind11Extension(
        "anygrad.tensor.tensor_c",
        [
            "anygrad/tensor/bind_tensor.cpp",
            "backend/anygrad/Th/ThAllocate.cpp",
            "backend/anygrad/Th/ThBaseops.cpp",
            "backend/anygrad/Th/Thhelpers.cpp",
            "backend/anygrad/Th/Thgemm.cpp",
            "backend/anygrad/Th/Tharrange.cpp"
        ],
        language="c++",
        extra_compile_args=compile_args
    ),
    Pybind11Extension(
        "anygrad.utils.utils_c",
        [
            "anygrad/utils/utils_bind.cpp",
            "backend/anygrad/utils/random_num.cpp",
            "backend/anygrad/Th/Thhelpers.cpp",
            "backend/anygrad/utils/init_ops.cpp",
            "backend/anygrad/utils/log_arithmetic.cpp",
        ],
        language="c++",
        extra_compile_args=compile_args
    )
]

setup(
    name="anygrad",
    version=version,
    description="A Tensor module that allows a deep learning framework to switch seamlessly between different engines.",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author="Ruhaan",
    author_email="ruhaan123dalal@gmail.com",
    license="Apache License",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    packages=find_packages(),
    package_dir={"": "."},
    package_data={
        "anygrad": ["Tensor/*.py", "__init__.py", "anygrad/*.py", "Tensor/utils/*.py"],
    },
    include_package_data=True,
    install_requires=[
        "pybind11"
    ]
)
