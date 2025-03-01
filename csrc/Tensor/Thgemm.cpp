#include <vector>
#include <utility>
#include <algorithm>

#include "Th.hpp"
#include "ThTypes.hpp"

#define vector_f32 std::vector<float_t>
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>

bool is_matmul_broadcast(vector_i32 shape1, vector_i32 shape2, int32_t dim1, int32_t dim2) {
    int max_dim = std::max(dim1, dim2);
    if (dim2 == 1) {
        if (shape1[dim1 - 1] != shape2[0])
            return false;
    } else if (shape1[dim1 - 1] != shape2[dim2 - 2]) {
        return false;
    }

    for (int i = 0; i < max_dim - 2; i++) {
        int new_dim1 = (i >= dim1 - 2) ? 1 : shape1[dim1 - 3 - i];
        int new_dim2 = (i >= dim2 - 2) ? 1 : shape2[dim2 - 3 - i];

        if (new_dim1 != 1 && new_dim2 != 1 && new_dim1 != new_dim2)
            return false;
    }
    return true;
}

vector_i32 matmul_broadcast_shape(vector_i32 shape1, vector_i32 shape2, int32_t dim1, int32_t dim2) {
    int max_dim = std::max(dim1, dim2);
    vector_i32 shape3(max_dim);

    for (int i = 0; i < max_dim - 2; i++) {
        int new_dim1 = (i >= dim1 - 2) ? 1 : shape1[dim1 - 3 - i];
        int new_dim2 = (i >= dim2 - 2) ? 1 : shape2[dim2 - 3 - i];
        if (max_dim - 3 - i >= 0) {  
            shape3[max_dim - 3 - i] = std::max(new_dim1, new_dim2);
        }
    }

    shape3[max_dim - 2] = shape1[dim1 - 2];
    shape3[max_dim - 1] = (dim2 == 1) ? 1 : shape2[dim2 - 1];
    return shape3;
}

template <typename U>
U matmul2d(const U& data1, const U& data2, int32_t I_shape, int32_t K_shape, int32_t J_shape) {
    const int32_t block_size = 256;
    U ans_data(I_shape * J_shape, 0);

    for (int ii = 0; ii < I_shape; ii += block_size) {
        for (int jj = 0; jj < J_shape; jj += block_size) {
            for (int kk = 0; kk < K_shape; kk += block_size) {
                for (int i = ii; i < std::min(ii + block_size, I_shape); ++i) {
                    for (int j = jj; j < std::min(jj + block_size, J_shape); ++j) {
                        float sum = 0.0f;
                        for (int k = kk; k < std::min(kk + block_size, K_shape); ++k) {
                            sum += data1[i * K_shape + k] * data2[k * J_shape + j];
                        }
                        ans_data[i * J_shape + j] += sum;
                    }
                }
            }
        }
    }
    return ans_data;
}

template <typename T, typename U>
std::pair<U, vector_i32> matmulNd(const T& tensor1, const T& tensor2) {
    vector_i32 ans_shape = matmul_broadcast_shape(tensor1.shape, tensor2.shape, tensor1.ndim, tensor2.ndim);
    int32_t ans_dim = ans_shape.size();
    int32_t size = calculate_size(ans_shape, ans_dim);
    int32_t max_dim = std::max(tensor1.ndim, tensor2.ndim);

    vector_i32 result_stride1 = broadcast_stride(tensor1.shape, tensor1.stride, tensor1.ndim, max_dim);
    vector_i32 result_stride2 = broadcast_stride(tensor2.shape, tensor2.stride, tensor2.ndim, max_dim);

    int32_t batch_size = 1;
    for (int i = 0; i < ans_dim - 2; i++) {
        batch_size *= ans_shape[i];
    }

    U result_data(size);
    
    int32_t I = ans_shape[ans_dim - 2];                 
    int32_t J = ans_shape[ans_dim - 1];                 
    int32_t K = tensor1.shape[tensor1.ndim - 1];        
    
    for (int b = 0; b < batch_size; b++) {
        U batch_data1(I * K);
        U batch_data2(K * J);
        
        for (int i = 0; i < I; i++) {
            for (int k = 0; k < K; k++) {
                size_t offset1 = 0;
                for (int d = 0; d < max_dim - 2; d++) {
                    int batch_idx = (b / (ans_shape[d+1] * ans_shape[d+2])) % ans_shape[d];
                    offset1 += batch_idx * result_stride1[d];
                }
                offset1 += i * tensor1.stride[tensor1.ndim - 2] + k * tensor1.stride[tensor1.ndim - 1];
                batch_data1[i * K + k] = tensor1.data[offset1];
            }
        }
        
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++) {
                size_t offset2 = 0;
                for (int d = 0; d < max_dim - 2; d++) {
                    int batch_idx = (b / (ans_shape[d+1] * ans_shape[d+2])) % ans_shape[d];
                    offset2 += batch_idx * result_stride2[d];
                }
                offset2 += k * tensor2.stride[tensor2.ndim - 2] + j * tensor2.stride[tensor2.ndim - 1];
                batch_data2[k * J + j] = tensor2.data[offset2];
            }
        }
        
        U batch_result = matmul2d(batch_data1, batch_data2, I, K, J);
        
        std::copy(batch_result.begin(), batch_result.end(), 
                 result_data.begin() + b * I * J);
    }

    return {result_data, ans_shape};
}
template <typename T>
T transpose2d(const T& src_mat, int32_t rows, int32_t cols) {
    T tgt_mat(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            tgt_mat[i + j * rows] = src_mat[i * cols + j];
        }
    }
    return tgt_mat;
}

template <typename T, typename U>
std::pair<U, vector_i32> transposeNd(const T& tensor1, int32_t dim0, int32_t dim1) {
    vector_i32 shape = tensor1.shape;
    std::swap(shape[dim0], shape[dim1]);
    
    int32_t total_size = calculate_size(shape, shape.size());
    U result_data(total_size);
    
    int32_t dim0_size = tensor1.shape[dim0];
    int32_t dim1_size = tensor1.shape[dim1];
    int32_t block_size = dim0_size * dim1_size;
    
    int32_t outer_size = 1;
    for (int i = 0; i < std::min(dim0, dim1); i++) {
        outer_size *= tensor1.shape[i];
    }
    
    int32_t inner_size = 1;
    for (int i = std::max(dim0, dim1) + 1; i < tensor1.ndim; i++) {
        inner_size *= tensor1.shape[i];
    }
    
    for (int outer = 0; outer < outer_size; outer++) {
        for (int inner = 0; inner < inner_size; inner++) {
            int32_t offset = outer * block_size * inner_size + inner * block_size;
            U block(tensor1.data.begin() + offset, 
                   tensor1.data.begin() + offset + block_size);
            U transposed = transpose2d(block, dim0_size, dim1_size);
            std::copy(transposed.begin(), transposed.end(), 
                     result_data.begin() + offset);
        }
    }
    
    return {result_data, shape};
}

std::pair<vector_f32, vector_i32> MatmulFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return matmulNd<FloatTensorBase, vector_f32>(tensor1, tensor2);
}

std::pair<vector_f64, vector_i32> MatmulFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return matmulNd<DoubleTensorBase, vector_f64>(tensor1, tensor2);
}

std::pair<vector_i32, vector_i32> MatmulInt32(Int32TensorBase tensor1, Int32TensorBase tensor2){
    return matmulNd<Int32TensorBase, vector_i32>(tensor1, tensor2);
}

std::pair<vector_i64, vector_i32> MatmulInt64(Int64TensorBase tensor1, Int64TensorBase tensor2){
    return matmulNd<Int64TensorBase, vector_i64>(tensor1, tensor2);
}

std::pair<vector_f32, vector_i32> TransFloat32(FloatTensorBase tensor, int32_t dim0, int32_t dim1){
    return transposeNd<FloatTensorBase, vector_f32>(tensor, dim0, dim1);
}

std::pair<vector_f64, vector_i32> TransFloat64(DoubleTensorBase tenosr, int32_t dim0, int32_t dim1){
    return transposeNd<DoubleTensorBase, vector_f64>(tenosr, dim0, dim1);
}

std::pair<vector_i32, vector_i32> TransInt32(Int32TensorBase tensor, int32_t dim0, int32_t dim1){
    return transposeNd<Int32TensorBase, vector_i32>(tensor, dim0, dim1);
}

std::pair<vector_i64, vector_i32> TransInt64(Int64TensorBase tensor, int32_t dim0, int32_t dim1){
    return transposeNd<Int64TensorBase, vector_i64>(tensor, dim0, dim1);
}

std::pair<vector_bool, vector_i32> TransBool(BoolTensorBase tensor, int32_t dim0, int32_t dim1){
    return transposeNd<BoolTensorBase, vector_bool>(tensor, dim0, dim1);
}