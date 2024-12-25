#include <vector>
#include <utility>

#include "../clib/ThTypes.hpp"
#include "../clib/Thallops.hpp"

#define vector_f32 std::vector<float_t>
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>

template <typename T>
std::pair<T, vector_i32> ZerosConfig(vector_i32 shape){
    T result_data;
    int32_t size = calculate_size(shape, shape.size());
    result_data.resize(size, 0);
    return {result_data, shape};
}

template <typename T, typename U>
std::pair<U, vector_i32> ZerosLikeConfig(T tensor){
    return ZerosConfig<U>(tensor.shape);
}

template <typename T>
std::pair<T, vector_i32> OnesConfig(vector_i32 shape){
    T result_data;
    int32_t size = calculate_size(shape, shape.size());
    result_data.resize(size, 1);
    return {result_data, shape};
}

template <typename T, typename U>
std::pair<U, vector_i32> OnesLikeConfig(T tensor){
    return OnesConfig<U>(tensor.shape);
}

std::pair<vector_f32, vector_i32> zerosFloat32(vector_i32 shape){
    return ZerosConfig<vector_f32>(shape);
}
std::pair<vector_f64, vector_i32> zerosFloat64(vector_i32 shape){
    return ZerosConfig<vector_f64>(shape);
}
std::pair<vector_f32, vector_i32> onesFloat32(vector_i32 shape){
    return OnesConfig<vector_f32>(shape);
}
std::pair<vector_f64, vector_i32> onesFloat64(vector_i32 shape){
    return OnesConfig<vector_f64>(shape);
}