#include <vector>
#include <utility>

#include "Thallops.hpp"
#include "ThTypes.hpp"

#define vector_f32 std::vector<float_t>
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>

bool is_matmul_broadcase(vector_i32 shape1, vector_i32 shape2, int32_t dim1, int32_t dim2){
    int max_dim = std::max(dim1, dim2);
    if (dim2 == 1){
        if (shape1[dim1 - 1] != shape2[0])
            return false;
        
    } else if (shape1[dim1 - 1] != shape2[dim2 - 2])
        return false;

    for (int i = 0; i < max_dim - 2; i++){
        int new_dim1 = (i >= dim1 - 2) ? 1: shape1[dim1 - 3 - i];
        int new_dim2 = (i >= dim2 - 2) ? 1: shape2[dim2 - 3 - i];

        if (new_dim1 != 1 && new_dim2 != 1 && new_dim1 != new_dim2)
            return false;

    }
    return true;
}


vector_i32 matmul_broadcast_shape(vector_i32 shape1, vector_i32 shape2, int32_t dim1, int32_t dim2){
    int max_dim = std::max(dim1, dim2);

    vector_i32 shape3(max_dim);

    for (int i = 0; i < max_dim - 2; i++){
        int new_dim1 = (i >= dim1 - 2) ? 1: shape1[dim1 - 3 - i];
        int new_dim2 = (i >= dim2 - 2) ? 1: shape2[dim2 - 3 - i];

        shape3[max_dim - 3 - i] = (new_dim1 > new_dim2) ? new_dim1 : new_dim2;
    }

    shape3[max_dim - 2] = shape1[dim1 - 2];
    shape3[max_dim - 1] = (dim2 == 1) ? 1 : shape2[dim2 - 1];

    return shape3;
}

