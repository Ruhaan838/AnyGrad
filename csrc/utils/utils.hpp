#ifndef UTILS_HPP
#define UTILS_HPP

#include "generator.hpp"
#include <utility>

//random_num.cpp
std::pair<vector_f32, vector_i32> randFloat32(vector_i32 shape, Generator *generator);
std::pair<vector_f64, vector_i32> randFloat64(vector_i32 shape, Generator *generator);

std::pair<vector_i32, vector_i32> randintInt32(vector_i32 shape, int32_t low, int32_t high, Generator *generator);
std::pair<vector_i64, vector_i32> randintInt64(vector_i32 shape, int32_t low, int32_t high, Generator *generator);

std::pair<vector_f32, vector_i32> zerosFloat32(vector_i32 shape);
std::pair<vector_f64, vector_i32> zerosFloat64(vector_i32 shape);
std::pair<vector_i32, vector_i32> zerosInt32(vector_i32 shape);
std::pair<vector_i64, vector_i32> zerosInt64(vector_i32 shape);

std::pair<vector_f32, vector_i32> onesFloat32(vector_i32 shape);
std::pair<vector_f64, vector_i32> onesFloat64(vector_i32 shape);
std::pair<vector_i32, vector_i32> onesInt32(vector_i32 shape);
std::pair<vector_i64, vector_i32> onesInt64(vector_i32 shape);

std::pair<vector_f32, vector_i32> LogFloat32(FloatTensorBase tensor1);
std::pair<vector_f64, vector_i32> LogFloat64(DoubleTensorBase tensor1);
std::pair<vector_i32, vector_i32> LogInt32(Int32TensorBase tensor1);
std::pair<vector_i64, vector_i32> LogInt64(Int64TensorBase tensor1);

std::pair<vector_f32, vector_i32> Log10Float32(FloatTensorBase tensor1);
std::pair<vector_f64, vector_i32> Log10Float64(DoubleTensorBase tensor1);
std::pair<vector_i32, vector_i32> Log10Int32(Int32TensorBase tensor1);
std::pair<vector_i64, vector_i32> Log10Int64(Int64TensorBase tensor1);

std::pair<vector_f32, vector_i32> Log2Float32(FloatTensorBase tensor1);
std::pair<vector_f64, vector_i32> Log2Float64(DoubleTensorBase tensor1);
std::pair<vector_i32, vector_i32> Log2Int32(Int32TensorBase tensor1);
std::pair<vector_i64, vector_i32> Log2Int64(Int64TensorBase tensor1);

std::pair<vector_f32, vector_i32> ExpFloat32(FloatTensorBase tensor1);
std::pair<vector_f64, vector_i32> ExpFloat64(DoubleTensorBase tensor1);
std::pair<vector_i32, vector_i32> ExpInt32(Int32TensorBase tensor1);
std::pair<vector_i64, vector_i32> ExpInt64(Int64TensorBase tensor1);

std::pair<vector_f32, vector_i32> Exp2Float32(FloatTensorBase tensor1);
std::pair<vector_f64, vector_i32> Exp2Float64(DoubleTensorBase tensor1);
std::pair<vector_i32, vector_i32> Exp2Int32(Int32TensorBase tensor1);
std::pair<vector_i64, vector_i32> Exp2Int64(Int64TensorBase tensor1);

#endif