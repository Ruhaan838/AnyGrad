#include <vector>
#include <utility>

#include "../Tensor/ThTypes.hpp"
#include "../Tensor/Th.hpp"

#define vector_f32 std::vector<float_t>
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>
#define vector_i64 std::vector<int64_t>

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

std::pair<vector_i32, vector_i32> LogInt32(Int32TensorBase tensor1){
    return LogConfig<Int32TensorBase, vector_i32, std::function<int32_t(int32_t)>> (tensor1, 
                                                                                    [](int32_t num) {return std::log(num);});
}

std::pair<vector_i64, vector_i32> LogInt64(Int64TensorBase tensor1){
    return LogConfig<Int64TensorBase, vector_i64, std::function<int64_t(int64_t)>> (tensor1, 
                                                                                    [](int64_t num) {return std::log(num);});
}

std::pair<vector_f32, vector_i32> Log10Float32(FloatTensorBase tensor1){
    return LogConfig<FloatTensorBase, vector_f32, std::function<float_t(float_t)>> (tensor1, 
                                                                                    [](float_t num) {return std::log10(num);});
}

std::pair<vector_f64, vector_i32> Log10Float64(DoubleTensorBase tensor1){
    return LogConfig<DoubleTensorBase, vector_f64, std::function<double_t(double_t)>> (tensor1, 
                                                                                    [](double_t num) {return std::log10(num);});
}

std::pair<vector_i32, vector_i32> Log10Int32(Int32TensorBase tensor1){
    return LogConfig<Int32TensorBase, vector_i32, std::function<int32_t(int32_t)>> (tensor1, 
        [](int32_t num) { return static_cast<int32_t>(std::log10(static_cast<double>(num))); });
}

std::pair<vector_i64, vector_i32> Log10Int64(Int64TensorBase tensor1){
    return LogConfig<Int64TensorBase, vector_i64, std::function<int64_t(int64_t)>> (tensor1, 
        [](int64_t num) { return static_cast<int64_t>(std::log10(static_cast<double>(num))); });
}

std::pair<vector_f32, vector_i32> Log2Float32(FloatTensorBase tensor1){
    return LogConfig<FloatTensorBase, vector_f32, std::function<float_t(float_t)>> (tensor1, 
                                                                                    [](float_t num) {return std::log2(num);});
}

std::pair<vector_f64, vector_i32> Log2Float64(DoubleTensorBase tensor1){
    return LogConfig<DoubleTensorBase, vector_f64, std::function<double_t(double_t)>> (tensor1, 
                                                                                    [](double_t num) {return std::log2(num);});
}

std::pair<vector_i32, vector_i32> Log2Int32(Int32TensorBase tensor1){
    return LogConfig<Int32TensorBase, vector_i32, std::function<int32_t(int32_t)>> (tensor1, 
        [](int32_t num) { return static_cast<int32_t>(std::log2(static_cast<double>(num))); });
}

std::pair<vector_i64, vector_i32> Log2Int64(Int64TensorBase tensor1){
    return LogConfig<Int64TensorBase, vector_i64, std::function<int64_t(int64_t)>> (tensor1, 
        [](int64_t num) { return static_cast<int64_t>(std::log2(static_cast<double>(num))); });
}

std::pair<vector_f32, vector_i32> ExpFloat32(FloatTensorBase tensor1){
    return LogConfig<FloatTensorBase, vector_f32, std::function<float_t(float_t)>> (tensor1, 
                                                                                    [](float_t num) {return std::exp(num);});
}

std::pair<vector_f64, vector_i32> ExpFloat64(DoubleTensorBase tensor1){
    return LogConfig<DoubleTensorBase, vector_f64, std::function<double_t(double_t)>> (tensor1, 
                                                                                    [](double_t num) {return std::exp(num);});
}

std::pair<vector_i32, vector_i32> ExpInt32(Int32TensorBase tensor1){
    return LogConfig<Int32TensorBase, vector_i32, std::function<int32_t(int32_t)>> (tensor1, 
        [](int32_t num) { return static_cast<int32_t>(std::exp(static_cast<double>(num))); });
}

std::pair<vector_i64, vector_i32> ExpInt64(Int64TensorBase tensor1){
    return LogConfig<Int64TensorBase, vector_i64, std::function<int64_t(int64_t)>> (tensor1, 
        [](int64_t num) { return static_cast<int64_t>(std::exp(static_cast<double>(num))); });
}

std::pair<vector_f32, vector_i32> Exp2Float32(FloatTensorBase tensor1){
    return LogConfig<FloatTensorBase, vector_f32, std::function<float_t(float_t)>> (tensor1, 
                                                                                    [](float_t num) {return std::exp2(num);});
}

std::pair<vector_f64, vector_i32> Exp2Float64(DoubleTensorBase tensor1){
    return LogConfig<DoubleTensorBase, vector_f64, std::function<double_t(double_t)>> (tensor1, 
                                                                                    [](double_t num) {return std::exp2(num);});
}

std::pair<vector_i32, vector_i32> Exp2Int32(Int32TensorBase tensor1){
    return LogConfig<Int32TensorBase, vector_i32, std::function<int32_t(int32_t)>> (tensor1, 
        [](int32_t num) { return static_cast<int32_t>(std::exp2(static_cast<double>(num))); });
}

std::pair<vector_i64, vector_i32> Exp2Int64(Int64TensorBase tensor1){
    return LogConfig<Int64TensorBase, vector_i64, std::function<int64_t(int64_t)>> (tensor1, 
        [](int64_t num) { return static_cast<int64_t>(std::exp2(static_cast<double>(num))); });
}