#include <vector>
#include <iostream>

#include "../OpsCenter.hpp"

#define vector_i32 std::vector<int32_t>

vector_i32 calculate_stride(vector_i32 shape, int32_t ndim){
    vector_i32 stride(ndim); 
    stride[ndim - 1] = 1;
    for(int i = ndim - 2; i >= 0; i--){
        stride[i] = stride[i+1] * shape[i+1];
    }
    return stride;
}

int32_t calculate_size(vector_i32 shape, int32_t ndim){
    int32_t size = 1;
    for(int32_t i = 0; i < ndim; i++)
        size *= shape[i];
    return size;
}

vector_i32 broadcast_stride(vector_i32 shape, vector_i32 stride, int32_t dim, int32_t max_dim){
    vector_i32 result_stride(max_dim); 
    for(int i = 0; i < max_dim; i++){
        int dim_a = (i >= dim) ? 1 : shape[dim - 1 - i];
        result_stride[max_dim - 1 - i] = (dim_a == 1) ? 0 : stride[dim - 1 - i];
    }
    return result_stride;
}

int32_t broadcast_shape(vector_i32 shape1, vector_i32 shape2, vector_i32 &result_shape, int32_t dim1, int32_t dim2, int32_t max_dim){
    result_shape.resize(max_dim);
    for(int i = 0; i < max_dim; i++){
        int32_t dim_a = (i >= dim1) ? 1 : shape1[dim1 - 1 - i];
        int32_t dim_b = (i >= dim2) ? 1 : shape2[dim2 - 1 - i];
        if (dim_a != 1 && dim_b != 1 && dim_a != dim_b)
            return -1;
        
        result_shape[max_dim - 1 - i] = (dim_a > dim_b) ? dim_a : dim_b;
    }
    return max_dim;
}

bool isbroadcast(vector_i32 shape1, vector_i32 shape2, int dim1, int dim2) {
    int max_dim = std::max(dim1, dim2);
    for (int i = 0; i < max_dim; i++) {
        int dim_a = (i >= dim1) ? 1 : shape1[dim1 - 1 - i];
        int dim_b = (i >= dim2) ? 1 : shape2[dim2 - 1 - i];
        if (dim_a != 1 && dim_b != 1 && dim_a != dim_b) {
            return false;
        }
    }
    return true; 
}

void update_offset(int32_t *offset1, int32_t *offset2, int32_t *n_idx, int32_t max_dim, vector_i32 stride, vector_i32 resut_stride1, vector_i32 resut_stride2){
    for (int i = 0; i < max_dim; i++){
        int stride_idx = *n_idx / stride[i];
        *n_idx %= stride[i];
        *offset1 += stride_idx * resut_stride1[i];
        *offset2 += stride_idx * resut_stride2[i];
    }
}

bool is_sum_allow(int32_t dim_to_sum, int32_t tensor_dim){
    if (dim_to_sum == -1)
        return true;
    if (dim_to_sum < -1 || dim_to_sum >= tensor_dim)
        return false;
    return true;
}
