// #include <utility>
// #include <functional>

// #include "grad_helper.hpp"
// #include "../../Tensor/clib/ThTypes.hpp"

// #include "../../Tensor/clib/Thallops.hpp"

// template <typename T>
// void GradFunctions<T>::add_grad(T tensor1, T tensor2, T ans_tensor){

//     auto _backward = [tensor1, tensor2, ans_tensor]() {
//         if (tensor1.requires_grad == true){
//             tensor1.grad += ans_tensor.grad;
//         }
//         if (tensor2.requires_grad == true){
//             tensor2.grad += ans_tensor.grad;
//         }
//     };

//     return _backward;
// }

// template <typename T>
// void GradFunctions<T>::sum_grad(T tensor, T ans_tensor){

//     auto _backward = [tensor, ans_tensor](){
//         if (tensor.requires_grad == true){
//             tensor.grad += ans_tensor.grad;
//         }
//     };
 
//     return _backward;
// }

// template <typename T>
// void add_grad(T tensor1, T tensor2, T ans_tensor){
//     auto _backward = [tensor1, tensor2, ans_tensor]() {
//         if (tensor1.requires_grad == true){
//             auto result = AddFloat32(tensor1, tensor2);
//         }
//         if (tensor2.requires_grad == true){
//             tensor2.grad += ans_tensor.grad;
//         }
//     };

//     return _backward;
// }

// void add_grad_32(FloatTensorBase tenosr1, FloatTensorBase tensor2, FloatTensorBase ans_tensor){
//     return add_grad<FloatTensorBase>(tenosr1, tensor2, ans_tensor);
// }