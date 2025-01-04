#include "../OpsCenter.hpp"
#include "../clib/ThTypes.hpp"
#include "generator.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(utils_c, msg){
    py::class_<Generator>(msg, "GeneratorBase")
        .def(py::init<int32_t>())
        .def("manual_seed", &Generator::manual_seed)
        ;

    msg.def("randFloat32", &randFloat32, py::return_value_policy::reference);
    msg.def("randFloat64", &randFloat64, py::return_value_policy::reference);

}

