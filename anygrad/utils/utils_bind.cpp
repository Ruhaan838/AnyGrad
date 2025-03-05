#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../csrc/Th/Th.hpp"
#include "../../csrc/utils/generator.hpp"
#include "../../csrc/utils/utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(utils_c, msg){
    py::class_<Generator>(msg, "GeneratorBase")
        .def(py::init<int32_t>())
        .def("manual_seed", &Generator::manual_seed)
        ;

    //rand
    msg.def("RandFloat32", &randFloat32);
    msg.def("RandFloat64", &randFloat64);

    //randint
    msg.def("RandintInt32", &randintInt32);
    msg.def("RandintInt64", &randintInt64);

    //initializer
    msg.def("ZerosFloat32", &zerosFloat32);
    msg.def("ZerosFloat64", &zerosFloat64);
    msg.def("ZerosInt32", &zerosInt32);
    msg.def("ZerosInt64", &zerosInt64);

    msg.def("OnesFloat32", &onesFloat32);
    msg.def("OnesFloat64", &onesFloat64);
    msg.def("OnesInt32", &onesInt32);
    msg.def("OnesInt64", &onesInt64);

    //log arithmetic
    msg.def("LogFloat32", &LogFloat32);
    msg.def("LogFloat64", &LogFloat64);

    msg.def("Log10Float32", &Log10Float32);
    msg.def("Log10Float64", &Log10Float64);

    msg.def("Log2Float32", &Log2Float32);
    msg.def("Log2Float64", &Log2Float64);

    msg.def("ExpFloat32", &ExpFloat32);
    msg.def("ExpFloat64", &ExpFloat64);

    msg.def("Exp2Float32", &Exp2Float32);
    msg.def("Exp2Float64", &Exp2Float64);
}

