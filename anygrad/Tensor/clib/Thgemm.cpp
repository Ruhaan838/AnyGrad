#include <vector>
#include <utility>

#include "../OpsCenter.hpp"
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
        shape3[max_dim - 3 - i] = std::max(new_dim1, new_dim2);
    }

    shape3[max_dim - 2] = shape1[dim1 - 2];
    shape3[max_dim - 1] = (dim2 == 1) ? 1 : shape2[dim2 - 1];
    return shape3;
}

template <typename U>
U matmul2d(U data1, U data2, int32_t I_shape, int32_t J_shape, int32_t K_shape) {
    int32_t block_size = 256;
    U ans_data(I_shape * J_shape, 0);  

    for (int ii = 0; ii < I_shape; ii += block_size) {
        for (int jj = 0; jj < J_shape; jj += block_size) {
            for (int kk = 0; kk < K_shape; kk += block_size) {

                for (int i = ii; i < std::min(ii + block_size, I_shape); i++) {
                    for (int k = kk; k < std::min(kk + block_size, K_shape); k++) {

                        auto temp = data1[i * K_shape + k];
                        for (int j = jj; j < std::min(jj + block_size, J_shape); j++) {
                            ans_data[i * J_shape + j] += temp * data2[k * J_shape + j];
                        }

                    }
                }

            }
        }
    }
    return ans_data;
}

template <typename T, typename U>
std::pair<U, vector_i32> matmulNd(T tensor1, T tensor2) {
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
    for (int i = 0; i < batch_size; i++) {
        auto st_indx1 = tensor1.data.begin() + i * result_stride1[0];
        U batch_data1(st_indx1, st_indx1 + tensor1.shape[tensor1.ndim - 1] * ans_shape[ans_dim - 2]);
        
        auto st_indx2 = tensor2.data.begin() + i * result_stride2[0];
        U batch_data2(st_indx2, st_indx2 + tensor2.shape[tensor2.ndim - 1] * ans_shape[ans_dim - 2]);
        
        U batch_result = matmul2d(batch_data1, batch_data2, 
                                 ans_shape[ans_dim - 2], 
                                 ans_shape[ans_dim - 1], 
                                 tensor1.shape[tensor1.ndim - 1]);

        std::copy(batch_result.begin(), batch_result.end(), 
                 result_data.begin() + i * ans_shape[ans_dim - 2] * ans_shape[ans_dim - 1]);
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

std::pair<vector_f32, vector_i32> TransFloat32(FloatTensorBase tensor, int32_t dim0, int32_t dim1){
    return transposeNd<FloatTensorBase, vector_f32>(tensor, dim0, dim1);
}

std::pair<vector_f64, vector_i32> TransFloat64(DoubleTensorBase tenosr, int32_t dim0, int32_t dim1){
    return transposeNd<DoubleTensorBase, vector_f64>(tenosr, dim0, dim1);
}