#ifndef THALLOPS_HPP
#define THALLOPS_HPP

#include <vector>
#include <utility>

#include "ThTypes.hpp"

#define vector_f32 std::vector<float_t> 
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>

//ThBaseopsF32.cpp
std::pair<vector_f32, vector_i32> AddFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
std::pair<vector_f32, vector_i32> SumFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims) ;

//ThBaseopsF64.cpp
std::pair<vector_f64, vector_i32> AddFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
std::pair<vector_f64, vector_i32> SumFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);

//Thhelpers.cpp
vector_i32 calculate_stride(vector_i32 shape, int32_t ndim);
int32_t calculate_size(vector_i32 shape, int32_t ndim);
vector_i32 broadcast_stride(vector_i32 shape, vector_i32 stride, int32_t dim, int32_t max_dim);
int32_t broadcast_shape(vector_i32 shape1, vector_i32 shape2, vector_i32 &result_shape, int32_t dim1, int32_t dim2, int32_t max_dim);
void update_offset(int32_t *offset1, int32_t *offset2, int32_t *n_idx, int32_t max_dim, vector_i32 stride, vector_i32 resut_stride1, vector_i32 resut_stride2);
bool isbroadcast(vector_i32 shape1, vector_i32 shape2, int32_t dim1, int32_t dim2);
bool is_sum_allow(int32_t dim_to_sum, int32_t tensor_dim);

//utils/anygrad_utils.cpp
std::pair<vector_f32, vector_i32> zerosFloat32(vector_i32 shape);
std::pair<vector_f64, vector_i32> zerosFloat64(vector_i32 shape);

std::pair<vector_f32, vector_i32> onesFloat32(vector_i32 shape);
std::pair<vector_f64, vector_i32> onesFloat64(vector_i32 shape);

#endif