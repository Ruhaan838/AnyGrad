#include <vector>
#include <utility>
#include <cassert>

#include "ThTypes.hpp"
#include "../OpsCenter.hpp"

#define vector_f32 std::vector<float_t> 
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>

template <typename T, typename U, typename Op>
std::pair<U, vector_i32> BaseConfigOp(T tensor1, T tensor2, Op op){

    U result_data;
    vector_i32 result_shape;
    int32_t max_dim = tensor1.ndim > tensor2.ndim ? tensor1.ndim: tensor2.ndim;

    vector_i32 result_stride1 = broadcast_stride(tensor1.shape, tensor1.stride, tensor1.ndim, max_dim);
    vector_i32 result_stride2 = broadcast_stride(tensor2.shape, tensor2.stride, tensor2.ndim, max_dim);

    int32_t allow = broadcast_shape(tensor1.shape, tensor2.shape, result_shape, tensor1.ndim, tensor2.ndim, max_dim);

    if (allow == -1){
        assert("The Shape is not broadcasted");
    }

    int32_t total_ele = calculate_size(result_shape, result_shape.size());

    vector_i32 result_stride = calculate_stride(result_shape, result_shape.size());

    result_data.resize(total_ele); 

    for(int32_t idx = 0; idx < total_ele; idx++){
        int32_t offset1 = 0; int32_t offset2 = 0;
        int n_idx = idx;

        update_offset(&offset1, &offset2, &n_idx, max_dim, result_stride, result_stride1, result_stride2);
        result_data[idx] = op(tensor1.data[offset1],tensor2.data[offset2]);
    }

    //I think this is the best way to delete the vector
    vector_i32().swap(result_stride1);
    vector_i32().swap(result_stride2);
    vector_i32().swap(result_stride);

    return {result_data, result_shape};
}

template <typename T, typename U>
std::pair<U, vector_i32> SumConfig(T tensor, int32_t dim_to_sum = -1, bool keepdims = false) {
    U result_data;
    vector_i32 result_shape;

    if (dim_to_sum == -1) {
        float sum = 0.0f;
        int32_t total_ele = calculate_size(tensor.shape, tensor.ndim);
        
        for (int32_t i = 0; i < total_ele; i++) {
            sum += tensor.data[i];
        }
        if (keepdims) {
            result_shape = vector_i32(tensor.ndim, 1);
        } else {
            result_shape = {1};
        }
        result_data = {sum};
        return {result_data, result_shape};
    }
    if (dim_to_sum < 0 || dim_to_sum >= tensor.ndim) {
        assert("Dimension out of range");
    }
    if (keepdims) {
        result_shape = tensor.shape;
        result_shape[dim_to_sum] = 1;
    } else {
        result_shape = tensor.shape;
        result_shape.erase(result_shape.begin() + dim_to_sum);
        if (result_shape.empty()) {
            result_shape.push_back(1);
        }
    }
    int32_t result_size = calculate_size(result_shape, result_shape.size());
    result_data.resize(result_size, 0.0f);

    int32_t total_ele = calculate_size(tensor.shape, tensor.ndim);
    for (int32_t idx = 0; idx < total_ele; idx++) {
        int32_t remaining = idx;
        int32_t result_idx = 0;
        vector_i32 coords(tensor.ndim);
        for (int32_t dim = tensor.ndim - 1; dim >= 0; dim--) {
            coords[dim] = remaining % tensor.shape[dim];
            remaining /= tensor.shape[dim];
        }
        int32_t result_dim = 0;
        for (int32_t dim = 0; dim < tensor.ndim; dim++) {
            if (dim == dim_to_sum) {
                continue;
            }
            result_idx += coords[dim] * calculate_stride(result_shape, result_shape.size())[result_dim];
            result_dim++;
        }
        result_data[result_idx] += tensor.data[idx];
    }
    return {result_data, result_shape};
}

std::pair<vector_f32, vector_i32> AddFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, std::function<float_t(float_t, float_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](float_t num1, float_t num2) {return num1 + num2;});
}

std::pair<vector_f64, vector_i32> AddFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, std::function<double_t(double_t, double_t)>>(tensor1, 
                                                                                                   tensor2, 
                                                                                                   [](double_t num1, double_t num2) {return num1 + num2;});   
}

std::pair<vector_f32, vector_i32> SumFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims) {
    return SumConfig<FloatTensorBase, vector_f32>(tensor, dim_to_sum, keepdims);
}

std::pair<vector_f64, vector_i32> SumFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims) {
    return SumConfig<DoubleTensorBase, vector_f64>(tensor, dim_to_sum, keepdims);
}

std::pair<vector_f32, vector_i32> MulFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, std::function<float_t(float_t, float_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](float_t num1, float_t num2) {return num1 * num2;});
}

std::pair<vector_f64, vector_i32> MulFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, std::function<double_t(double_t, double_t)>>(tensor1, 
                                                                                                   tensor2, 
                                                                                                   [](double_t num1, double_t num2) {return num1 * num2;});   
}

std::pair<vector_f32, vector_i32> SubFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, std::function<float_t(float_t, float_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](float_t num1, float_t num2) {return num1 - num2;});
}

std::pair<vector_f64, vector_i32> SubFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, std::function<double_t(double_t, double_t)>>(tensor1, 
                                                                                                   tensor2, 
                                                                                                   [](double_t num1, double_t num2) {return num1 - num2;});   
}

std::pair<vector_f32, vector_i32> DivFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, std::function<float_t(float_t, float_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](float_t num1, float_t num2) {return num1 / num2;});
}

std::pair<vector_f64, vector_i32> DivFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, std::function<double_t(double_t, double_t)>>(tensor1, 
                                                                                                   tensor2, 
                                                                                                   [](double_t num1, double_t num2) {return num1 / num2;});   
}

std::pair<vector_f32, vector_i32> PowFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, std::function<float_t(float_t, float_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](float_t num1, float_t num2) {return std::pow(num1, num2);});
}

std::pair<vector_f64, vector_i32> PowFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, std::function<double_t(double_t, double_t)>>(tensor1, 
                                                                                                   tensor2, 
                                                                                                   [](double_t num1, double_t num2) {return std::pow(num1, num2);});   
}