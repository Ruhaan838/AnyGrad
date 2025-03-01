#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../csrc/Tensor/Th.hpp"
#include "../../csrc/utils/generator.hpp"
#include "../../csrc/utils/utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(utils_c, msg){
    py::class_<Generator>(msg, "GeneratorBase")
        .def(py::init<int32_t>())
        .def("manual_seed", &Generator::manual_seed)
        ;

    //rand
    msg.def("RandFloat32", &randFloat32, py::return_value_policy::reference);
    msg.def("RandFloat64", &randFloat64, py::return_value_policy::reference);

    //randint
    msg.def("RandintInt32", &randintInt32, py::return_value_policy::reference);
    msg.def("RandintInt64", &randintInt64, py::return_value_policy::reference);

    //initializer
    msg.def("ZerosFloat32", &zerosFloat32, py::return_value_policy::reference);
    msg.def("ZerosFloat64", &zerosFloat64, py::return_value_policy::reference);
    msg.def("ZerosInt32", &zerosInt32, py::return_value_policy::reference);
    msg.def("ZerosInt64", &zerosInt64, py::return_value_policy::reference);

    msg.def("OnesFloat32", &onesFloat32, py::return_value_policy::reference);
    msg.def("OnesFloat64", &onesFloat64, py::return_value_policy::reference);
    msg.def("OnesInt32", &onesInt32, py::return_value_policy::reference);
    msg.def("OnesInt64", &onesInt64, py::return_value_policy::reference);

    //log arithmetic
    msg.def("LogFloat32", &LogFloat32, py::return_value_policy::reference);
    msg.def("LogFloat64", &LogFloat64, py::return_value_policy::reference);

    msg.def("Log10Float32", &Log10Float32, py::return_value_policy::reference);
    msg.def("Log10Float64", &Log10Float64, py::return_value_policy::reference);

    msg.def("Log2Float32", &Log2Float32, py::return_value_policy::reference);
    msg.def("Log2Float64", &Log2Float64, py::return_value_policy::reference);

    msg.def("ExpFloat32", &ExpFloat32, py::return_value_policy::reference);
    msg.def("ExpFloat64", &ExpFloat64, py::return_value_policy::reference);

    msg.def("Exp2Float32", &Exp2Float32, py::return_value_policy::reference);
    msg.def("Exp2Float64", &Exp2Float64, py::return_value_policy::reference);
}

