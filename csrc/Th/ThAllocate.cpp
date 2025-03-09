#include <vector>

#include "ThTypes.hpp"
#include "Th.hpp"

#define vector_f32 vector<float>
#define vector_f64 vector<double>
#define vector_i32 vector<int32_t>
#define vector_i64 vector<int64_t>
#define vector_bool vector<bool>

using namespace std;

BaseTensor::BaseTensor(vector_i16 shape){
    this->shape = shape; 
    this->ndim = shape.size(); 
    this->stride = calculate_stride(shape, shape.size());
    this->size = calculate_size(shape, shape.size());
};

FloatTensorBase::FloatTensorBase(vector_f32 data, vector_i16 shape) : BaseTensor(shape){
    this->data = data; 
    this->dtype = "float32";
}

DoubleTensorBase::DoubleTensorBase(vector_f64 data, vector_i16 shape) : BaseTensor(shape){
    this->data = data;
    this->dtype = "float64";
}

Int32TensorBase::Int32TensorBase(vector_i32 data, vector_i16 shape) : BaseTensor(shape){
    this->data = data;
    this->dtype = "int32";
}

Int64TensorBase::Int64TensorBase(vector_i64 data, vector_i16 shape) : BaseTensor(shape){
    this->data = data;
    this->dtype = "int64";
}

BoolTensorBase::BoolTensorBase(vector_bool data, vector_i16 shape) : BaseTensor(shape){
    this->data = data;
    this->dtype = "bool";
}