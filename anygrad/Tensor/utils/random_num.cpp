#include <vector>
#include <utility>
#include <string>
#include <random>

#include "../clib/ThTypes.hpp"
#include "../OpsCenter.hpp"
#include "generator.hpp"

#define vector_f32 std::vector<float_t>
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>

template <typename U, typename T>
std::pair<U, vector_i32> randConfig(vector_i32 shape, Generator* generator = nullptr){
    U result_data;

    //local engine
    static std::mt19937 global_engine(std::random_device{}());
    static std::uniform_real_distribution<T> gloabal_dist(0.0, 1.0);

    int32_t size = calculate_size(shape, shape.size());
    result_data.resize(size, 0); //initalize the data
    for (int32_t i = 0; i < size; i++){
        if(generator)
            result_data[i] = generator->randfloat();
        else
            result_data[i] = gloabal_dist(global_engine);
    }

    return {result_data, shape};
}

std::pair<vector_f32, vector_i32> randFloat32(vector_i32 shape, Generator generator){
    return randConfig<vector_f32, float_t>(shape, &generator);
}
std::pair<vector_f64, vector_i32> randFloat64(vector_i32 shape, Generator generator){
    return randConfig<vector_f64, double_t>(shape, &generator);
}
