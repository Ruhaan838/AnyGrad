#ifndef THALLOPS_HPP
#define THALLOPS_HPP

#include <vector>
#include <utility>

#include "clib/ThTypes.hpp"
#include "utils/generator.hpp"

#define vector_f32 std::vector<float_t> 
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>

//ThBaseopsF32.cpp
std::pair<vector_f32, vector_i32> AddFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
std::pair<vector_f32, vector_i32> SumFloat32(FloatTensorBase tensor, int32_t dim_to_sum, bool keepdims) ;
std::pair<vector_f32, vector_i32> MulFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
std::pair<vector_f32, vector_i32> SubFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
std::pair<vector_f32, vector_i32> DivFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
std::pair<vector_f32, vector_i32> PowFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);


//ThBaseopsF64.cpp
std::pair<vector_f64, vector_i32> AddFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
std::pair<vector_f64, vector_i32> SumFloat64(DoubleTensorBase tensor, int32_t dim_to_sum, bool keepdims);
std::pair<vector_f64, vector_i32> MulFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
std::pair<vector_f64, vector_i32> SubFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
std::pair<vector_f64, vector_i32> DivFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
std::pair<vector_f64, vector_i32> PowFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);


//Thhelpers.cpp
vector_i32 calculate_stride(vector_i32 shape, int32_t ndim);
int32_t calculate_size(vector_i32 shape, int32_t ndim);
vector_i32 broadcast_stride(vector_i32 shape, vector_i32 stride, int32_t dim, int32_t max_dim);
int32_t broadcast_shape(vector_i32 shape1, vector_i32 shape2, vector_i32 &result_shape, int32_t dim1, int32_t dim2, int32_t max_dim);
void update_offset(int32_t *offset1, int32_t *offset2, int32_t *n_idx, int32_t max_dim, vector_i32 stride, vector_i32 resut_stride1, vector_i32 resut_stride2);
bool isbroadcast(vector_i32 shape1, vector_i32 shape2, int dim1, int dim2);
bool is_sum_allow(int32_t dim_to_sum, int32_t tensor_dim);

//utils/anygrad_utils.cpp
std::pair<vector_f32, vector_i32> zerosFloat32(vector_i32 shape);
std::pair<vector_f64, vector_i32> zerosFloat64(vector_i32 shape);

std::pair<vector_f32, vector_i32> onesFloat32(vector_i32 shape);
std::pair<vector_f64, vector_i32> onesFloat64(vector_i32 shape);

std::pair<vector_f32, vector_i32> LogFloat32(FloatTensorBase tensor1);
std::pair<vector_f64, vector_i32> LogFloat64(DoubleTensorBase tensor1);

std::pair<vector_f32, vector_i32> Log10Float32(FloatTensorBase tensor1);
std::pair<vector_f64, vector_i32> Log10Float64(DoubleTensorBase tensor1);

std::pair<vector_f32, vector_i32> Log2Float32(FloatTensorBase tensor1);
std::pair<vector_f64, vector_i32> Log2Float64(DoubleTensorBase tensor1);

std::pair<vector_f32, vector_i32> ExpFloat32(FloatTensorBase tensor1);
std::pair<vector_f64, vector_i32> ExpFloat64(DoubleTensorBase tensor1);

std::pair<vector_f32, vector_i32> Exp2Float32(FloatTensorBase tensor1);
std::pair<vector_f64, vector_i32> Exp2Float64(DoubleTensorBase tensor1);

//gemm.cpp
vector_i32 matmul_broadcast_shape(vector_i32 shape1, vector_i32 shape2, int32_t dim1, int32_t dim2);
bool is_matmul_broadcast(vector_i32 shape1, vector_i32 shape2, int32_t dim1, int32_t dim2);
std::pair<vector_f32, vector_i32> MatmulFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2);
std::pair<vector_f64, vector_i32> MatmulFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2);
std::pair<vector_f32, vector_i32> TransFloat32(FloatTensorBase tensor, int32_t dim0, int32_t dim1);
std::pair<vector_f64, vector_i32> TransFloat64(DoubleTensorBase tenosr, int32_t dim0, int32_t dim1);

//random_num.cpp
std::pair<vector_f32, vector_i32> randFloat32(vector_i32 shape, Generator generator);
std::pair<vector_f64, vector_i32> randFloat64(vector_i32 shape, Generator generator);


#endif