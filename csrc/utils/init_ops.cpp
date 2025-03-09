#include <vector>
#include <utility>

#include "../Th/ThTypes.hpp"
#include "../Th/Th.hpp"

#define vector_f32 vector<float>
#define vector_f64 vector<double>
#define vector_i16 vector<int16_t>
#define vector_i32 vector<int32_t>
#define vector_i64 vector<int64_t>
#define vector_bool vector<bool>

template <typename T>
pair<T, vector_i16> ZerosOrOnesConfig(vector_i16 shape, int32_t num){
    T result_data;
    int32_t size = calculate_size(shape, shape.size());
    result_data.resize(size, num);
    return {result_data, shape};
}

pair<vector_f32, vector_i16> zerosFloat32(vector_i16 shape){
    return ZerosOrOnesConfig<vector_f32>(shape, 0);
}
pair<vector_f64, vector_i16> zerosFloat64(vector_i16 shape){
    return ZerosOrOnesConfig<vector_f64>(shape, 0);
}
pair<vector_i32, vector_i16> zerosInt32(vector_i16 shape){
    return ZerosOrOnesConfig<vector_i32>(shape, 0);
}
pair<vector_i64, vector_i16> zerosInt64(vector_i16 shape){
    return ZerosOrOnesConfig<vector_i64>(shape, 0);
}
pair<vector_f32, vector_i16> onesFloat32(vector_i16 shape){
    return ZerosOrOnesConfig<vector_f32>(shape, 1);
}
pair<vector_f64, vector_i16> onesFloat64(vector_i16 shape){
    return ZerosOrOnesConfig<vector_f64>(shape, 1);
}
pair<vector_i32, vector_i16> onesInt32(vector_i16 shape){
    return ZerosOrOnesConfig<vector_i32>(shape, 1);
}
pair<vector_i64, vector_i16> onesInt64(vector_i16 shape){
    return ZerosOrOnesConfig<vector_i64>(shape, 1);
}