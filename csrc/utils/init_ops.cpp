#include <vector>
#include <utility>

#include "../Tensor/ThTypes.hpp"
#include "../Tensor/Th.hpp"

#define vector_f32 std::vector<float_t>
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>
#define vector_i64 std::vector<int64_t>

template <typename T>
std::pair<T, vector_i32> ZerosOrOnesConfig(vector_i32 shape, int32_t num){
    T result_data;
    int32_t size = calculate_size(shape, shape.size());
    result_data.resize(size, num);
    return {result_data, shape};
}

std::pair<vector_f32, vector_i32> zerosFloat32(vector_i32 shape){
    return ZerosOrOnesConfig<vector_f32>(shape, 0);
}
std::pair<vector_f64, vector_i32> zerosFloat64(vector_i32 shape){
    return ZerosOrOnesConfig<vector_f64>(shape, 0);
}
std::pair<vector_i32, vector_i32> zerosInt32(vector_i32 shape){
    return ZerosOrOnesConfig<vector_i32>(shape, 0);
}
std::pair<vector_i64, vector_i32> zerosInt64(vector_i32 shape){
    return ZerosOrOnesConfig<vector_i64>(shape, 0);
}
std::pair<vector_f32, vector_i32> onesFloat32(vector_i32 shape){
    return ZerosOrOnesConfig<vector_f32>(shape, 1);
}
std::pair<vector_f64, vector_i32> onesFloat64(vector_i32 shape){
    return ZerosOrOnesConfig<vector_f64>(shape, 1);
}
std::pair<vector_i32, vector_i32> onesInt32(vector_i32 shape){
    return ZerosOrOnesConfig<vector_i32>(shape, 1);
}
std::pair<vector_i64, vector_i32> onesInt64(vector_i32 shape){
    return ZerosOrOnesConfig<vector_i64>(shape, 1);
}