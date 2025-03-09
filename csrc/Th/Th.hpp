#ifndef THALLOPS_HPP
#define THALLOPS_HPP

#include <vector>
#include <utility>

#include "ThTypes.hpp"

using namespace std;

#define vector_f32 vector<float>
#define vector_f64 vector<double>
#define vector_i16 vector<int16_t>
#define vector_i32 vector<int32_t>
#define vector_i64 vector<int64_t>
#define vector_bool vector<bool>

//ThBaseopsF32.cpp
pair<vector_f32, vector_i16> AddFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
pair<vector_f32, vector_i16> MulFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
pair<vector_f32, vector_i16> SubFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
pair<vector_f32, vector_i16> DivFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
pair<vector_f32, vector_i16> PowFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);

pair<vector_f32, vector_i16> SumFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f32, vector_i16> MeanFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f32, vector_i16> MinFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f32, vector_i16> MaxFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f32, vector_i16> MedianFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims);

//ThBaseopsF64.cpp
pair<vector_f64, vector_i16> AddFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
pair<vector_f64, vector_i16> MulFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
pair<vector_f64, vector_i16> SubFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
pair<vector_f64, vector_i16> DivFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
pair<vector_f64, vector_i16> PowFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);

pair<vector_f64, vector_i16> SumFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f64, vector_i16> MeanFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f64, vector_i16> MinFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f64, vector_i16> MaxFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f64, vector_i16> MedianFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);

//ThBaseopsI32.cpp
pair<vector_i32, vector_i16> AddInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);
pair<vector_i32, vector_i16> MulInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);
pair<vector_i32, vector_i16> SubInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);
pair<vector_f32, vector_i16> DivInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);
pair<vector_i32, vector_i16> PowInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);

pair<vector_f32, vector_i16> SumInt32(Int32TensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f32, vector_i16> MeanInt32(Int32TensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f32, vector_i16> MinInt32(Int32TensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f32, vector_i16> MaxInt32(Int32TensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f32, vector_i16> MedianInt32(Int32TensorBase tensor, int32_t dim_to_sum, bool keepdims);

//ThBaseopsI64.cpp
pair<vector_i64, vector_i16> AddInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);
pair<vector_i64, vector_i16> MulInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);
pair<vector_i64, vector_i16> SubInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);
pair<vector_f64, vector_i16> DivInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);
pair<vector_i64, vector_i16> PowInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);

pair<vector_f64, vector_i16> SumInt64(Int64TensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f64, vector_i16> MeanInt64(Int64TensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f64, vector_i16> MinInt64(Int64TensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f64, vector_i16> MaxInt64(Int64TensorBase tensor, int32_t dim_to_sum, bool keepdims);
pair<vector_f64, vector_i16> MedianInt64(Int64TensorBase tensor, int32_t dim_to_sum, bool keepdims);

//ThBaseopsBool.cpp
pair<vector_bool, vector_i16> AddBool(BoolTensorBase tensor1, BoolTensorBase tensor2);
pair<vector_bool, vector_i16> MulBool(BoolTensorBase tensor1, BoolTensorBase tensor2);
pair<vector_bool, vector_i16> SubBool(BoolTensorBase tensor1, BoolTensorBase tensor2);
pair<vector_bool, vector_i16> DivBool(BoolTensorBase tensor1, BoolTensorBase tensor2);

//Thhelpers.cpp
vector_i16 calculate_stride(vector_i16 shape, int32_t ndim);
int32_t calculate_size(vector_i16 shape, int32_t ndim);
vector_i16 broadcast_stride(vector_i16 shape, vector_i16 stride, int32_t dim, int32_t max_dim);
vector_i16 broadcast_shape(vector_i16 shape1, vector_i16 shape2, int32_t dim1, int32_t dim2, int32_t max_dim);
void update_offset(int32_t *offset1, int32_t *offset2, int32_t *n_idx, int32_t max_dim, vector_i16 stride, vector_i16 resut_stride1, vector_i16 resut_stride2);
bool isbroadcast(vector_i16 shape1, vector_i16 shape2, int dim1, int dim2);
bool is_sum_allow(int32_t dim_to_sum, int32_t tensor_dim);

//gemm.cpp
vector_i16 matmul_broadcast_shape(vector_i16 shape1, vector_i16 shape2, int32_t dim1, int32_t dim2);
bool is_matmul_broadcast(vector_i16 shape1, vector_i16 shape2, int32_t dim1, int32_t dim2);
pair<vector_f32, vector_i16> MatmulFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
pair<vector_f64, vector_i16> MatmulFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
pair<vector_i32, vector_i16> MatmulInt32(Int32TensorBase tensor1, Int32TensorBase tensor2);
pair<vector_i64, vector_i16> MatmulInt64(Int64TensorBase tensor1, Int64TensorBase tensor2);
pair<vector_f32, vector_i16> TransFloat32(FloatTensorBase tensor, int32_t dim0, int32_t dim1);
pair<vector_f64, vector_i16> TransFloat64(DoubleTensorBase tenosr, int32_t dim0, int32_t dim1);
pair<vector_i32, vector_i16> TransInt32(Int32TensorBase tensor, int32_t dim0, int32_t dim1);
pair<vector_i64, vector_i16> TransInt64(Int64TensorBase tenosr, int32_t dim0, int32_t dim1);
pair<vector_bool, vector_i16> TransBool(BoolTensorBase tensor, int32_t dim0, int32_t dim1);



#endif