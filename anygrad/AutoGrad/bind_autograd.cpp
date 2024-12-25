// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

// #include "clib/grad_helper.hpp"


// namespace py = pybind11;

// PYBIND11_MODULE(autograd_c, m) {
//     // py::class_<GradFunctions_f32>(m, "Grad_f32")
//     //     .def("add_grad", &GradFunctions_f32::add_grad)
//     //     .def("sum_grad", &GradFunctions_f32::sum_grad)
//     //     ;

//     // py::class_<GradFunctions_f64>(m, "Grad_f64")
//     //     .def("add_grad", &GradFunctions_f64::add_grad)
//     //     .def("sum_grad", &GradFunctions_f64::sum_grad)
//     //     ;
//     // m.def("add_grad", &add_grad, py::return_value_policy::reference);
// }