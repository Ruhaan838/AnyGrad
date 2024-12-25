#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "clib/ThTypes.hpp"
#include "clib/Thallops.hpp"

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
    
    msg.def("AddFloat32", &AddFloat32, py::return_value_policy::reference);
    msg.def("AddFloat64", &AddFloat64, py::return_value_policy::reference);
    msg.def("SumFloat32", &SumFloat32, py::return_value_policy::reference);
    msg.def("SumFloat64", &SumFloat64, py::return_value_policy::reference);

    msg.def("isbroadcast", &isbroadcast, py::return_value_policy::reference);
    msg.def("is_sum_allow", &is_sum_allow, py::return_value_policy::reference);

    msg.def("zerosfloat32", &zerosFloat32, py::return_value_policy::reference);
    msg.def("zerosfloat64", &zerosFloat64, py::return_value_policy::reference);

    msg.def("onesfloat32", &onesFloat32, py::return_value_policy::reference);
    msg.def("onesfloat64", &onesFloat64, py::return_value_policy::reference);

    // msg.def("DEBUG_64", &DEBUG_64, py::return_value_policy::reference);
}
