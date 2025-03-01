#ifndef THALLOPS_HPP
#define THALLOPS_HPP

#include <vector>
#include <utility>

#include "ThTypes.hpp"

#define vector_f32 std::vector<float_t> 
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>
#define vector_i64 std::vector<int64_t>
#define vector_bool std::vector<bool>

//ThBaseopsF32.cpp
std::pair<vector_f32, vector_i32> AddFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
std::pair<vector_f32, vector_i32> MulFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
std::pair<vector_f32, vector_i32> SubFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
std::pair<vector_f32, vector_i32> DivFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
std::pair<vector_f32, vector_i32> PowFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);

std::pair<vector_f32, vector_i32> SumFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f32, vector_i32> MeanFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f32, vector_i32> MinFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f32, vector_i32> MaxFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f32, vector_i32> MedianFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims);

//ThBaseopsF64.cpp
std::pair<vector_f64, vector_i32> AddFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
std::pair<vector_f64, vector_i32> MulFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
std::pair<vector_f64, vector_i32> SubFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
std::pair<vector_f64, vector_i32> DivFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
std::pair<vector_f64, vector_i32> PowFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);

std::pair<vector_f64, vector_i32> SumFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f64, vector_i32> MeanFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f64, vector_i32> MinFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f64, vector_i32> MaxFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f64, vector_i32> MedianFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);

//ThBaseopsI32.cpp
std::pair<vector_i32, vector_i32> AddInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);
std::pair<vector_i32, vector_i32> MulInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);
std::pair<vector_i32, vector_i32> SubInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);
std::pair<vector_i32, vector_i32> DivInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);
std::pair<vector_i32, vector_i32> PowInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);

std::pair<vector_f32, vector_i32> SumInt32(Int32TensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f32, vector_i32> MeanInt32(Int32TensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f32, vector_i32> MinInt32(Int32TensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f32, vector_i32> MaxInt32(Int32TensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f32, vector_i32> MedianInt32(Int32TensorBase tensor, int32_t dim_to_sum, bool keepdims);

//ThBaseopsI64.cpp
std::pair<vector_i64, vector_i32> AddInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);
std::pair<vector_i64, vector_i32> MulInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);
std::pair<vector_i64, vector_i32> SubInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);
std::pair<vector_i64, vector_i32> DivInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);
std::pair<vector_i64, vector_i32> PowInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);

std::pair<vector_f64, vector_i32> SumInt64(Int64TensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f64, vector_i32> MeanInt64(Int64TensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f64, vector_i32> MinInt64(Int64TensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f64, vector_i32> MaxInt64(Int64TensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f64, vector_i32> MedianInt64(Int64TensorBase tensor, int32_t dim_to_sum, bool keepdims);

//ThBaseopsBool.cpp
std::pair<vector_bool, vector_i32> AddBool(BoolTensorBase tensor1, BoolTensorBase tensor2);
std::pair<vector_bool, vector_i32> MulBool(BoolTensorBase tensor1, BoolTensorBase tensor2);
std::pair<vector_bool, vector_i32> SubBool(BoolTensorBase tensor1, BoolTensorBase tensor2);
std::pair<vector_bool, vector_i32> DivBool(BoolTensorBase tensor1, BoolTensorBase tensor2);

//Thhelpers.cpp
vector_i32 calculate_stride(vector_i32 shape, int32_t ndim);
int32_t calculate_size(vector_i32 shape, int32_t ndim);
vector_i32 broadcast_stride(vector_i32 shape, vector_i32 stride, int32_t dim, int32_t max_dim);
int32_t broadcast_shape(vector_i32 shape1, vector_i32 shape2, vector_i32 &result_shape, int32_t dim1, int32_t dim2, int32_t max_dim);
void update_offset(int32_t *offset1, int32_t *offset2, int32_t *n_idx, int32_t max_dim, vector_i32 stride, vector_i32 resut_stride1, vector_i32 resut_stride2);
bool isbroadcast(vector_i32 shape1, vector_i32 shape2, int dim1, int dim2);
bool is_sum_allow(int32_t dim_to_sum, int32_t tensor_dim);

//gemm.cpp
vector_i32 matmul_broadcast_shape(vector_i32 shape1, vector_i32 shape2, int32_t dim1, int32_t dim2);
bool is_matmul_broadcast(vector_i32 shape1, vector_i32 shape2, int32_t dim1, int32_t dim2);
std::pair<vector_f32, vector_i32> MatmulFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
std::pair<vector_f64, vector_i32> MatmulFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
std::pair<vector_i32, vector_i32> MatmulInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);
std::pair<vector_i64, vector_i32> MatmulInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);
std::pair<vector_f32, vector_i32> TransFloat32(FloatTensorBase tensor, int32_t dim0, int32_t dim1);
std::pair<vector_f64, vector_i32> TransFloat64(DoubleTensorBase tenosr, int32_t dim0, int32_t dim1);
std::pair<vector_i32, vector_i32> TransInt32(Int32TensorBase tensor, int32_t dim0, int32_t dim1);
std::pair<vector_i64, vector_i32> TransInt64(Int64TensorBase tenosr, int32_t dim0, int32_t dim1);

#endif