#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "clib/ThTypes.hpp"
#include "OpsCenter.hpp"

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
    msg.def("AddFloat32", &AddFloat32, py::return_value_policy::reference);
    msg.def("AddFloat64", &AddFloat64, py::return_value_policy::reference);
    msg.def("AddInt32", &AddInt32, py::return_value_policy::reference);
    msg.def("AddInt64", &AddInt64, py::return_value_policy::reference);
    msg.def("AddBool", &AddBool, py::return_value_policy::reference);

    msg.def("SubFloat32", &SubFloat32, py::return_value_policy::reference);
    msg.def("SubFloat64", &SubFloat64, py::return_value_policy::reference);
    msg.def("SubInt32", &SubInt32, py::return_value_policy::reference);
    msg.def("SubInt64", &SubInt64, py::return_value_policy::reference);
    msg.def("SubBool", &SubBool, py::return_value_policy::reference);
    
    msg.def("MulFloat32", &MulFloat32, py::return_value_policy::reference);
    msg.def("MulFloat64", &MulFloat64, py::return_value_policy::reference);
    msg.def("MulInt32", &MulInt32, py::return_value_policy::reference);
    msg.def("MulInt64", &MulInt64, py::return_value_policy::reference);
    msg.def("MulBool", &MulBool, py::return_value_policy::reference);

    msg.def("DivFloat32", &DivFloat32, py::return_value_policy::reference);
    msg.def("DivFloat64", &DivFloat64, py::return_value_policy::reference);
    msg.def("DivInt32", &DivInt32, py::return_value_policy::reference);
    msg.def("DivInt64", &DivInt64, py::return_value_policy::reference);
    msg.def("DivBool", &DivBool, py::return_value_policy::reference);

    msg.def("PowFloat32", &PowFloat32, py::return_value_policy::reference);
    msg.def("PowFloat64", &PowFloat64, py::return_value_policy::reference);
    msg.def("PowInt32", &PowInt32, py::return_value_policy::reference);
    msg.def("PowInt64", &PowInt64, py::return_value_policy::reference);

    //sums, means
    msg.def("SumFloat32", &SumFloat32, py::return_value_policy::reference);
    msg.def("SumFloat64", &SumFloat64, py::return_value_policy::reference);
    msg.def("SumInt32", &SumInt32, py::return_value_policy::reference);
    msg.def("SumInt64", &SumInt64, py::return_value_policy::reference);

    msg.def("MeanFloat32", &MeanFloat32, py::return_value_policy::reference);
    msg.def("MeanFloat64", &MeanFloat64, py::return_value_policy::reference);
    msg.def("MeanInt32", &MeanInt32, py::return_value_policy::reference);
    msg.def("MeanInt64", &MeanInt64, py::return_value_policy::reference);

    msg.def("MaxFloat32", &MaxFloat32, py::return_value_policy::reference);
    msg.def("MaxFloat64", &MaxFloat64, py::return_value_policy::reference);
    msg.def("MaxInt32", &MaxInt32, py::return_value_policy::reference);
    msg.def("MaxInt64", &MaxInt64, py::return_value_policy::reference);

    msg.def("MinFloat32", &MinFloat32, py::return_value_policy::reference);
    msg.def("MinFloat64", &MinFloat64, py::return_value_policy::reference);
    msg.def("MinInt32", &MinInt32, py::return_value_policy::reference);
    msg.def("MinInt64", &MinInt64, py::return_value_policy::reference);

    msg.def("MedianFloat32", &MedianFloat32, py::return_value_policy::reference);
    msg.def("MedianFloat64", &MedianFloat64, py::return_value_policy::reference);
    msg.def("MedianInt32", &MedianInt32, py::return_value_policy::reference);
    msg.def("MedianInt64", &MedianInt64, py::return_value_policy::reference);

    //rules
    msg.def("isbroadcast", &isbroadcast, py::return_value_policy::reference);
    msg.def("is_sum_allow", &is_sum_allow, py::return_value_policy::reference);

    //gemm
    msg.def("MatmulFloat32", &MatmulFloat32, py::return_value_policy::reference);
    msg.def("MatmulFloat64", &MatmulFloat64, py::return_value_policy::reference);
    msg.def("TransFloat32", &TransFloat32, py::return_value_policy::reference);
    msg.def("TransFloat64", &TransFloat64, py::return_value_policy::reference);
    msg.def("is_matmul_broadcast", &is_matmul_broadcast, py::return_value_policy::reference);
    
    // msg.def("DEBUG_64", &DEBUG_64, py::return_value_policy::reference);
}
