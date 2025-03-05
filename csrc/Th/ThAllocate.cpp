#include <vector>

#include "ThTypes.hpp"
#include "Th.hpp"

#define vector_f32 std::vector<float_t>
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>
#define vector_i64 std::vector<int64_t>
#define vector_bool std::vector<bool>

BaseTensor::BaseTensor(vector_i32 shape){
    this->shape = shape; 
    this->ndim = shape.size(); 
    this->stride = calculate_stride(shape, shape.size());
    this->size = calculate_size(shape, shape.size());
};

FloatTensorBase::FloatTensorBase(vector_f32 data, vector_i32 shape) : BaseTensor(shape){
    this->data = data; 
    this->dtype = "float32";
}

DoubleTensorBase::DoubleTensorBase(vector_f64 data, vector_i32 shape) : BaseTensor(shape){
    this->data = data;
    this->dtype = "float64";
}

Int32TensorBase::Int32TensorBase(vector_i32 data, vector_i32 shape) : BaseTensor(shape){
    this->data = data;
    this->dtype = "int32";
}

Int64TensorBase::Int64TensorBase(vector_i64 data, vector_i32 shape) : BaseTensor(shape){
    this->data = data;
    this->dtype = "int64";
}

BoolTensorBase::BoolTensorBase(vector_bool data, vector_i32 shape) : BaseTensor(shape){
    this->data = data;
    this->dtype = "bool";
}