#ifndef THTYPES_HPP
#define THTYPES_HPP

#include <vector>
#include <utility>
#include <functional>
#include <set>
#include <string>

#define vector_i32 std::vector<int32_t>
#define vector_f32 std::vector<float_t>
#define vector_f64 std::vector<double_t>

class BaseTensor{
    public:
        vector_i32 shape;
        vector_i32 stride;
        int32_t ndim;
        int32_t size;
        BaseTensor(vector_i32 shape);
};

class FloatTensorBase : public BaseTensor{
    public:
        vector_f32 data;
        std::string dtype;
        FloatTensorBase(vector_f32 data, vector_i32 shape);
};

class DoubleTensorBase : public BaseTensor{
    public:
        vector_f64 data;
        std::string dtype;
        DoubleTensorBase(vector_f64 data, vector_i32 shape);
};

// New tensor types:
class Int32TensorBase : public BaseTensor {
    public:
        std::vector<int32_t> data;
        std::string dtype;
        Int32TensorBase(std::vector<int32_t> data, vector_i32 shape);
};

class Int64TensorBase : public BaseTensor {
    public:
        std::vector<int64_t> data;
        std::string dtype;
        Int64TensorBase(std::vector<int64_t> data, vector_i32 shape);
};

class BoolTensorBase : public BaseTensor {
    public:
        std::vector<bool> data;
        std::string dtype;
        BoolTensorBase(std::vector<bool> data, vector_i32 shape);
};

#endif