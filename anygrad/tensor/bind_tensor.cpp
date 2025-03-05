#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../csrc/Th/Th.hpp"

#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(tensor_c, msg) {
    py::class_<FloatTensorBase>(msg, "float32")
        .def(py::init<std::vector<float_t>, std::vector<int32_t>>())
        .def_readonly("data", &FloatTensorBase::data)
        .def_readonly("shape", &FloatTensorBase::shape)
        .def_readonly("ndim", &FloatTensorBase::ndim)
        .def_readonly("dtype", &FloatTensorBase::dtype)
        .def_readonly("size", &FloatTensorBase::size)
        ;
        
    py::class_<DoubleTensorBase>(msg, "float64")
        .def(py::init<std::vector<double_t>, std::vector<int32_t>>())
        .def_readonly("data", &DoubleTensorBase::data)
        .def_readonly("shape", &DoubleTensorBase::shape)
        .def_readonly("ndim", &DoubleTensorBase::ndim)
        .def_readonly("dtype", &DoubleTensorBase::dtype)
        .def_readonly("size", &DoubleTensorBase::size)
        ;

    py::class_<Int32TensorBase>(msg, "int32")
        .def(py::init<std::vector<int32_t>, std::vector<int32_t>>())
        .def_readonly("data", &Int32TensorBase::data)
        .def_readonly("shape", &Int32TensorBase::shape)
        .def_readonly("ndim", &Int32TensorBase::ndim)
        .def_readonly("dtype", &Int32TensorBase::dtype)
        .def_readonly("size", &Int32TensorBase::size)
        ;

    py::class_<Int64TensorBase>(msg, "int64")
        .def(py::init<std::vector<int64_t>, std::vector<int32_t>>())
        .def_readonly("data", &Int64TensorBase::data)
        .def_readonly("shape", &Int64TensorBase::shape)
        .def_readonly("ndim", &Int64TensorBase::ndim)
        .def_readonly("dtype", &Int64TensorBase::dtype)
        .def_readonly("size", &Int64TensorBase::size)
        ;

    py::class_<BoolTensorBase>(msg, "bool")
        .def(py::init<std::vector<bool>, std::vector<int32_t>>())
        .def_readonly("data", &BoolTensorBase::data)
        .def_readonly("shape", &BoolTensorBase::shape)
        .def_readonly("ndim", &BoolTensorBase::ndim)
        .def_readonly("dtype", &BoolTensorBase::dtype)
        .def_readonly("size", &BoolTensorBase::size)
        ;
    
    //arithmetic
    msg.def("AddFloat32", &AddFloat32);
    msg.def("AddFloat64", &AddFloat64);
    msg.def("AddInt32", &AddInt32);
    msg.def("AddInt64", &AddInt64);
    msg.def("AddBool", &AddBool);

    msg.def("SubFloat32", &SubFloat32);
    msg.def("SubFloat64", &SubFloat64);
    msg.def("SubInt32", &SubInt32);
    msg.def("SubInt64", &SubInt64);
    msg.def("SubBool", &SubBool);
    
    msg.def("MulFloat32", &MulFloat32);
    msg.def("MulFloat64", &MulFloat64);
    msg.def("MulInt32", &MulInt32);
    msg.def("MulInt64", &MulInt64);
    msg.def("MulBool", &MulBool);

    msg.def("DivFloat32", &DivFloat32);
    msg.def("DivFloat64", &DivFloat64);
    msg.def("DivInt32", &DivInt32);
    msg.def("DivInt64", &DivInt64);
    msg.def("DivBool", &DivBool);

    msg.def("PowFloat32", &PowFloat32);
    msg.def("PowFloat64", &PowFloat64);
    msg.def("PowInt32", &PowInt32);
    msg.def("PowInt64", &PowInt64);

    //sums, means
    msg.def("SumFloat32", &SumFloat32);
    msg.def("SumFloat64", &SumFloat64);
    msg.def("SumInt32", &SumInt32);
    msg.def("SumInt64", &SumInt64);

    msg.def("MeanFloat32", &MeanFloat32);
    msg.def("MeanFloat64", &MeanFloat64);
    msg.def("MeanInt32", &MeanInt32);
    msg.def("MeanInt64", &MeanInt64);

    msg.def("MaxFloat32", &MaxFloat32);
    msg.def("MaxFloat64", &MaxFloat64);
    msg.def("MaxInt32", &MaxInt32);
    msg.def("MaxInt64", &MaxInt64);

    msg.def("MinFloat32", &MinFloat32);
    msg.def("MinFloat64", &MinFloat64);
    msg.def("MinInt32", &MinInt32);
    msg.def("MinInt64", &MinInt64);

    msg.def("MedianFloat32", &MedianFloat32);
    msg.def("MedianFloat64", &MedianFloat64);
    msg.def("MedianInt32", &MedianInt32);
    msg.def("MedianInt64", &MedianInt64);

    //rules
    msg.def("isbroadcast", &isbroadcast);
    msg.def("is_sum_allow", &is_sum_allow);

    //gemm
    msg.def("MatmulFloat32", &MatmulFloat32);
    msg.def("MatmulFloat64", &MatmulFloat64);
    msg.def("MatmulInt32", &MatmulInt32);
    msg.def("MatmulInt64", &MatmulInt64);
    msg.def("TransFloat32", &TransFloat32);
    msg.def("TransFloat64", &TransFloat64);
    msg.def("TransInt32", &TransInt32);
    msg.def("TransInt64", &TransInt64);
    msg.def("TransBool", &TransBool);
    msg.def("is_matmul_broadcast", &is_matmul_broadcast);
    
    // msg.def("DEBUG_64", &DEBUG_64);
}
