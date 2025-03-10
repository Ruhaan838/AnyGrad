#include <algorithm>
#include <limits>
#include <vector>
#include <utility>

#include "ThTypes.hpp"
#include "Th.hpp"

using namespace std;

enum class Ops { SUM, MEAN, MEDIAN, MIN, MAX };
template <typename U, typename V>
pair<V, vector_i16> ReduceConfig(U& tensor, Ops op, int32_t dim = -1, bool keepdims = false) {
    V result_data;
    vector_i16 result_shape;
    int32_t total_ele = tensor.size;

    using T = typename V::value_type;
    
    if (dim == -1) {
        // in this we init the value by 0 if it's SUM, MEAN, MEDIAN and otherwise
        // for MIN it's infinity
        // for MAX it's -infinity
        T init_val = (op == Ops::MIN) ? numeric_limits<T>::infinity() : 
                     (op == Ops::MAX) ? -numeric_limits<T>::infinity() : 0;
        
        result_data.resize(1, init_val); // then initalize the data with that value for size 1
        
        vector<T> vals; // median vector to get that collect the all terms 
        // then we can use this vals in median caclulation
        // odd numbers = vals[(n+1) / 2]
        // even numbers = vals[(n/2+1)] + vals[(n/2)] / 2
        
        for (int32_t i = 0; i < total_ele; i++) {
            T val = static_cast<T>(tensor.data[i]);
            if (op == Ops::MEDIAN) {
                vals.push_back(val);
            } else if (op == Ops::SUM || op == Ops::MEAN) {
                result_data[0] += val;
            } else if (op == Ops::MIN) {
                result_data[0] = min(result_data[0], val);
            } else if (op == Ops::MAX) {
                result_data[0] = max(result_data[0], val);
            }
        }
        
        if (op == Ops::MEAN) result_data[0] /= total_ele;
        else if (op == Ops::MEDIAN) {
            sort(vals.begin(), vals.end());
            int32_t temp = total_ele / 2;
            if (total_ele % 2 == 0)
                result_data[0] = (vals[temp - 1] + vals[temp]) / 2;
            else
                result_data[0] = vals[temp];
        }
        
        result_shape = keepdims ? vector_i16(tensor.ndim, 1) : vector_i16{1};
        return {result_data, result_shape};
    }
    
    if (keepdims) {
        result_shape = tensor.shape;
        result_shape[dim] = 1;
    } else {
        //this condition is handle the case when we have the tensor shape of 3D after reduction it's can be become the 2D tensor
        // so accourding to we need to change the shape
        result_shape.reserve(tensor.ndim - 1);
        for (int32_t i = 0; i < tensor.ndim; i++)
            if (i != dim) result_shape.push_back(tensor.shape[i]);
        if (result_shape.empty()) result_shape.push_back(1);
    }
    
    int32_t result_size = calculate_size(result_shape, result_shape.size());
    int32_t reduce_dim_size = tensor.shape[dim];
    result_data.resize(result_size);
    
    if (op != Ops::MEDIAN) {
        T init_val = (op == Ops::MIN) ? numeric_limits<T>::infinity() : 
                    (op == Ops::MAX) ? -numeric_limits<T>::infinity() : 0;
        fill(result_data.begin(), result_data.end(), init_val);
        // now this time we have the vector so we inti the hole vector with init_val
    }
    
    vector_i16 out_stride = calculate_stride(result_shape, result_shape.size());
    
    vector<vector<T>> median_val;
    if (op == Ops::MEDIAN) {
        median_val.resize(result_size);
        for (auto& v : median_val) v.reserve(reduce_dim_size);
    }
    
    //go thorugh the total_elements and calculate the op.
    for (int32_t idx = 0; idx < total_ele; idx++) {
        int32_t ind = 0;
        for (int32_t d = 0, rd = 0; d < tensor.ndim; d++) {
            if (d == dim) continue;
            int32_t offset = (idx / tensor.stride[d]) % tensor.shape[d];
            ind += offset * out_stride[rd++];
        }
        
        T value = static_cast<T>(tensor.data[idx]);
        
        if (op == Ops::MEDIAN) {
            median_val[ind].push_back(value);
        } else if (op == Ops::SUM || op == Ops::MEAN) {
            result_data[ind] += value;
        } else if (op == Ops::MIN) {
            result_data[ind] = min(result_data[ind], value);
        } else if (op == Ops::MAX) {
            result_data[ind] = max(result_data[ind], value);
        }
    }
    
    if (op == Ops::MEAN) {
        for (int32_t i = 0; i < result_size; i++)
            result_data[i] /= reduce_dim_size;
    } else if (op == Ops::MEDIAN) {
        for (int32_t i = 0; i < result_size; i++) {
            auto& vals = median_val[i];
            sort(vals.begin(), vals.end());
            if (vals.size() % 2 == 0)
                result_data[i] = (vals[vals.size()/2 - 1] + vals[vals.size()/2]) / 2;
            else
                result_data[i] = vals[vals.size() / 2];
        }
    }
    
    return {result_data, result_shape};
}

template <typename T, typename U, typename Op>
pair<U, vector_i16> BaseConfigOp(T tensor1, T tensor2, Op op){
    // for scaler tensor
    if (tensor2.size == 1){
        U result_data(tensor1.size, 0);
        for (int i = 0; i < tensor1.size; i ++){
            result_data[i] = op(tensor1.data[i], tensor2.data[0]);
        }
        return {result_data, tensor1.shape};
    }

    int32_t max_dim = max(tensor1.ndim, tensor2.ndim);

    vector_i16 result_stride1 = broadcast_stride(tensor1.shape, tensor1.stride, tensor1.ndim, max_dim);
    vector_i16 result_stride2 = broadcast_stride(tensor2.shape, tensor2.stride, tensor2.ndim, max_dim);

    vector_i16 result_shape = broadcast_shape(tensor1.shape, tensor2.shape, tensor1.ndim, tensor2.ndim, max_dim);
    int32_t total_ele = calculate_size(result_shape, result_shape.size());
    vector_i16 result_stride = calculate_stride(result_shape, result_shape.size());

    U result_data(total_ele);

    for(int32_t idx = 0; idx < total_ele; idx++){
        int32_t offset1 = 0; int32_t offset2 = 0;
        int n_idx = idx;

        update_offset(&offset1, &offset2, &n_idx, max_dim, result_stride, result_stride1, result_stride2);
        result_data[idx] = op(tensor1.data[offset1],tensor2.data[offset2]);
    }

    //I think this is the best way to delete the vector
    vector_i16().swap(result_stride1);
    vector_i16().swap(result_stride2);
    vector_i16().swap(result_stride);

    return {result_data, result_shape};
}

//arithmetic ops

//addition
pair<vector_f32, vector_i16> AddFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, function<float(float, float)>>(tensor1,tensor2, 
    [](float num1, float num2) {return num1 + num2;});
}

pair<vector_f64, vector_i16> AddFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, function<double(double, double)>>(tensor1,tensor2, 
    [](double num1, double num2) {return num1 + num2;});   
}

pair<vector_i32, vector_i16> AddInt32(Int32TensorBase tensor1, Int32TensorBase tensor2){
    return BaseConfigOp<Int32TensorBase, vector_i32, function<int32_t(int32_t, int32_t)>>(tensor1,tensor2, 
    [](int32_t num1, int32_t num2) {return num1 + num2;});
}

pair<vector_i64, vector_i16> AddInt64(Int64TensorBase tensor1, Int64TensorBase tensor2){
    return BaseConfigOp<Int64TensorBase, vector_i64, function<int64_t(int64_t, int64_t)>>(tensor1,tensor2, 
    [](int64_t num1, int64_t num2) {return num1 + num2;});
}

pair<vector_bool, vector_i16> AddBool(BoolTensorBase tensor1, BoolTensorBase tensor2){
    return BaseConfigOp<BoolTensorBase, vector_bool, function<bool(bool, bool)>>(tensor1,tensor2, 
    [](bool num1, bool num2) {return num1 + num2;});
}


//multiplication
pair<vector_f32, vector_i16> MulFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, function<float(float, float)>>(tensor1,tensor2, 
    [](float num1, float num2) {return num1 * num2;});
}

pair<vector_f64, vector_i16> MulFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, function<double(double, double)>>(tensor1, tensor2, 
    [](double num1, double num2) {return num1 * num2;});   
}
    
pair<vector_i32, vector_i16> MulInt32(Int32TensorBase tensor1, Int32TensorBase tensor2){
    return BaseConfigOp<Int32TensorBase, vector_i32, function<int32_t(int32_t, int32_t)>>(tensor1, tensor2, 
    [](int32_t num1, int32_t num2) {return num1 * num2;});
}

pair<vector_i64, vector_i16> MulInt64(Int64TensorBase tensor1, Int64TensorBase tensor2){
    return BaseConfigOp<Int64TensorBase, vector_i64, function<int64_t(int64_t, int64_t)>>(tensor1, tensor2, 
    [](int64_t num1, int64_t num2) {return num1 * num2;});
}

pair<vector_bool, vector_i16> MulBool(BoolTensorBase tensor1, BoolTensorBase tensor2){
    return BaseConfigOp<BoolTensorBase, vector_bool, function<bool(bool, bool)>>(tensor1, tensor2,
    [](bool num1, bool num2) {return num1 * num2;});
}


pair<vector_f32, vector_i16> SubFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, function<float(float, float)>>(tensor1, tensor2, 
    [](float num1, float num2) {return num1 - num2;});
}

pair<vector_f64, vector_i16> SubFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, function<double(double, double)>>(tensor1, tensor2, 
    [](double num1, double num2) {return num1 - num2;});   
}

pair<vector_i32, vector_i16> SubInt32(Int32TensorBase tensor1, Int32TensorBase tensor2){
    return BaseConfigOp<Int32TensorBase, vector_i32, function<int32_t(int32_t, int32_t)>>(tensor1, tensor2, 
    [](int32_t num1, int32_t num2) {return num1 - num2;});
}

pair<vector_i64, vector_i16> SubInt64(Int64TensorBase tensor1, Int64TensorBase tensor2){
    return BaseConfigOp<Int64TensorBase, vector_i64, function<int64_t(int64_t, int64_t)>>(tensor1, tensor2, 
    [](int64_t num1, int64_t num2) {return num1 - num2;});
}

pair<vector_bool, vector_i16> SubBool(BoolTensorBase tensor1, BoolTensorBase tensor2){
    return BaseConfigOp<BoolTensorBase, vector_bool, function<bool(bool, bool)>>(tensor1, tensor2,
    [](bool num1, bool num2) {return num1 - num2;});
}


pair<vector_f32, vector_i16> DivFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, function<float(float, float)>>(tensor1, tensor2, 
    [](float num1, float num2) {return num1 / num2;});
}

pair<vector_f64, vector_i16> DivFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, function<double(double, double)>>(tensor1, tensor2, 
    [](double num1, double num2) {return num1 / num2;});   
}

pair<vector_f32, vector_i16> DivInt32(Int32TensorBase tensor1, Int32TensorBase tensor2){
    return BaseConfigOp<Int32TensorBase, vector_f32, function<float(int32_t, int32_t)>>(tensor1, tensor2, 
    [](int32_t num1, int32_t num2) {return static_cast<float>(num1) / static_cast<float>(num2);});
}

pair<vector_f64, vector_i16> DivInt64(Int64TensorBase tensor1, Int64TensorBase tensor2){
    return BaseConfigOp<Int64TensorBase, vector_f64, function<double(int64_t, int64_t)>>(tensor1, tensor2, 
    [](int64_t num1, int64_t num2) {return static_cast<double>(num1) / static_cast<double>(num2);});
}

pair<vector_bool, vector_i16> DivBool(BoolTensorBase tensor1, BoolTensorBase tensor2){
    return BaseConfigOp<BoolTensorBase, vector_bool, function<bool(bool, bool)>>(tensor1, tensor2,
    [](bool num1, bool num2) {return num1 / num2;});
}


pair<vector_f32, vector_i16> PowFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, function<float(float, float)>>(tensor1, tensor2, 
    [](float num1, float num2) {return pow(num1, num2);});
}

pair<vector_f64, vector_i16> PowFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, function<double(double, double)>>(tensor1, tensor2, 
    [](double num1, double num2) {return pow(num1, num2);});   
}

pair<vector_i32, vector_i16> PowInt32(Int32TensorBase tensor1, Int32TensorBase tensor2){
    return BaseConfigOp<Int32TensorBase, vector_i32, function<int32_t(int32_t, int32_t)>>(tensor1, tensor2, 
    [](int32_t num1, int32_t num2) {return pow(num1, num2);});
}

pair<vector_i64, vector_i16> PowInt64(Int64TensorBase tensor1, Int64TensorBase tensor2){
    return BaseConfigOp<Int64TensorBase, vector_i64, function<int64_t(int64_t, int64_t)>>(tensor1, tensor2, 
    [](int64_t num1, int64_t num2) {return pow(num1, num2);});
}


// Reduction ops
pair<vector_f32, vector_i16> SumFloat32(FloatTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<FloatTensorBase, vector_f32>(tensor, Ops::SUM, dim, keepdims);
}

pair<vector_f64, vector_i16> SumFloat64(DoubleTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<DoubleTensorBase, vector_f64>(tensor, Ops::SUM, dim, keepdims);
}

pair<vector_f32, vector_i16> SumInt32(Int32TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int32TensorBase, vector_f32>(tensor, Ops::SUM, dim, keepdims);
}

pair<vector_f64, vector_i16> SumInt64(Int64TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int64TensorBase, vector_f64>(tensor, Ops::SUM, dim, keepdims);
}


pair<vector_f32, vector_i16> MeanFloat32(FloatTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<FloatTensorBase, vector_f32>(tensor, Ops::MEAN, dim, keepdims);
}

pair<vector_f64, vector_i16> MeanFloat64(DoubleTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<DoubleTensorBase, vector_f64>(tensor, Ops::MEAN, dim, keepdims);
}

pair<vector_f32, vector_i16> MeanInt32(Int32TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int32TensorBase, vector_f32>(tensor, Ops::MEAN, dim, keepdims);
}

pair<vector_f64, vector_i16> MeanInt64(Int64TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int64TensorBase, vector_f64>(tensor, Ops::MEAN, dim, keepdims);
}

pair<vector_f32, vector_i16> MedianFloat32(FloatTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<FloatTensorBase, vector_f32>(tensor, Ops::MEDIAN, dim, keepdims);
}

pair<vector_f64, vector_i16> MedianFloat64(DoubleTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<DoubleTensorBase, vector_f64>(tensor, Ops::MEDIAN, dim, keepdims);
}

pair<vector_f32, vector_i16> MedianInt32(Int32TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int32TensorBase, vector_f32>(tensor, Ops::MEDIAN, dim, keepdims);
}  

pair<vector_f64, vector_i16> MedianInt64(Int64TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int64TensorBase, vector_f64>(tensor, Ops::MEDIAN, dim, keepdims);
}

pair<vector_f32, vector_i16> MinFloat32(FloatTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<FloatTensorBase, vector_f32>(tensor, Ops::MIN, dim, keepdims);
}

pair<vector_f64, vector_i16> MinFloat64(DoubleTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<DoubleTensorBase, vector_f64>(tensor, Ops::MIN, dim, keepdims);
}

pair<vector_f32, vector_i16> MinInt32(Int32TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int32TensorBase, vector_f32>(tensor, Ops::MIN, dim, keepdims);
}

pair<vector_f64, vector_i16> MinInt64(Int64TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int64TensorBase, vector_f64>(tensor, Ops::MIN, dim, keepdims);
}

pair<vector_f32, vector_i16> MaxFloat32(FloatTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<FloatTensorBase, vector_f32>(tensor, Ops::MAX, dim, keepdims);
}

pair<vector_f64, vector_i16> MaxFloat64(DoubleTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<DoubleTensorBase, vector_f64>(tensor, Ops::MAX, dim, keepdims);
}

pair<vector_f32, vector_i16> MaxInt32(Int32TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int32TensorBase, vector_f32>(tensor, Ops::MAX, dim, keepdims);
}

pair<vector_f64, vector_i16> MaxInt64(Int64TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int64TensorBase, vector_f64>(tensor, Ops::MAX, dim, keepdims);
}