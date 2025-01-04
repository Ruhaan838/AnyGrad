#include <vector>
#include <utility>

#include "../clib/ThTypes.hpp"
#include "../OpsCenter.hpp"

#define vector_f32 std::vector<float_t>
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>

template <typename T, typename U, typename Op>
std::pair<U, vector_i32> LogConfig(T tensor1, Op op){
    U result_data(tensor1.size);
    for(int32_t i = 0; i < tensor1.size; i++){
        result_data[i] = op(tensor1.data[i]);
    }
    return {result_data, tensor1.shape};
}

std::pair<vector_f32, vector_i32> LogFloat32(FloatTensorBase tensor1){
    return LogConfig<FloatTensorBase, vector_f32, std::function<float_t(float_t)>> (tensor1, 
                                                                                    [](float_t num) {return std::log(num);});
}

std::pair<vector_f64, vector_i32> LogFloat64(DoubleTensorBase tensor1){
    return LogConfig<DoubleTensorBase, vector_f64, std::function<double_t(double_t)>> (tensor1, 
                                                                                    [](double_t num) {return std::log(num);});
}

std::pair<vector_f32, vector_i32> Log10Float32(FloatTensorBase tensor1){
    return LogConfig<FloatTensorBase, vector_f32, std::function<float_t(float_t)>> (tensor1, 
                                                                                    [](float_t num) {return std::log10(num);});
}

std::pair<vector_f64, vector_i32> Log10Float64(DoubleTensorBase tensor1){
    return LogConfig<DoubleTensorBase, vector_f64, std::function<double_t(double_t)>> (tensor1, 
                                                                                    [](double_t num) {return std::log10(num);});
}

std::pair<vector_f32, vector_i32> Log2Float32(FloatTensorBase tensor1){
    return LogConfig<FloatTensorBase, vector_f32, std::function<float_t(float_t)>> (tensor1, 
                                                                                    [](float_t num) {return std::log2(num);});
}

std::pair<vector_f64, vector_i32> Log2Float64(DoubleTensorBase tensor1){
    return LogConfig<DoubleTensorBase, vector_f64, std::function<double_t(double_t)>> (tensor1, 
                                                                                    [](double_t num) {return std::log2(num);});
}

std::pair<vector_f32, vector_i32> ExpFloat32(FloatTensorBase tensor1){
    return LogConfig<FloatTensorBase, vector_f32, std::function<float_t(float_t)>> (tensor1, 
                                                                                    [](float_t num) {return std::exp(num);});
}

std::pair<vector_f64, vector_i32> ExpFloat64(DoubleTensorBase tensor1){
    return LogConfig<DoubleTensorBase, vector_f64, std::function<double_t(double_t)>> (tensor1, 
                                                                                    [](double_t num) {return std::exp(num);});
}

std::pair<vector_f32, vector_i32> Exp2Float32(FloatTensorBase tensor1){
    return LogConfig<FloatTensorBase, vector_f32, std::function<float_t(float_t)>> (tensor1, 
                                                                                    [](float_t num) {return std::exp2(num);});
}

std::pair<vector_f64, vector_i32> Exp2Float64(DoubleTensorBase tensor1){
    return LogConfig<DoubleTensorBase, vector_f64, std::function<double_t(double_t)>> (tensor1, 
                                                                                    [](double_t num) {return std::exp2(num);});
}