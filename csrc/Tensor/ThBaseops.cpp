#include <algorithm>
#include <limits>
#include <vector>
#include <utility>
#include <cassert>

#include "ThTypes.hpp"
#include "Th.hpp"

#define vector_f32 std::vector<float_t> 
#define vector_f64 std::vector<double_t>
#define vector_i32 std::vector<int32_t>
#define vector_i64 std::vector<int64_t>
#define vector_bool std::vector<bool>

enum class ReductionOp { SUM, MEAN, MEDIAN, MIN, MAX };

template <typename TensorType, typename ResultType>
std::pair<ResultType, vector_i32> ReduceConfig(const TensorType& tensor, ReductionOp op, int32_t dim = -1, bool keepdims = false) {
    static_assert(std::is_floating_point<typename ResultType::value_type>::value, 
                 "ResultType must be a vector of floating point values");
    
    ResultType result_data;
    vector_i32 result_shape;
    const int32_t total_ele = calculate_size(tensor.shape, tensor.ndim);

    if (total_ele == 0) {
        throw std::runtime_error("Cannot reduce empty tensor");
    }

    if (dim == -1) {
        using T = typename ResultType::value_type;
        T init_val;
        
        switch(op) {
            case ReductionOp::SUM:
            case ReductionOp::MEAN:
                init_val = 0;
                break;
            case ReductionOp::MIN:
                init_val = std::numeric_limits<T>::infinity();
                break;
            case ReductionOp::MAX:
                init_val = -std::numeric_limits<T>::infinity();
                break;
            case ReductionOp::MEDIAN:
                init_val = 0;
                break;
        }

        result_data.resize(1, init_val);

        switch(op) {
            case ReductionOp::SUM:
            case ReductionOp::MEAN: {
                for (int32_t i = 0; i < total_ele; i++) {
                    result_data[0] += static_cast<T>(tensor.data[i]);
                }
                if (op == ReductionOp::MEAN) {
                    result_data[0] /= total_ele;
                }
                break;
            }
            case ReductionOp::MIN: {
                for (int32_t i = 0; i < total_ele; i++) {
                    result_data[0] = std::min(result_data[0], static_cast<T>(tensor.data[i]));
                }
                break;
            }
            case ReductionOp::MAX: {
                for (int32_t i = 0; i < total_ele; i++) {
                    result_data[0] = std::max(result_data[0], static_cast<T>(tensor.data[i]));
                }
                break;
            }
            case ReductionOp::MEDIAN: {
                std::vector<T> temp;
                temp.reserve(total_ele);
                for (int32_t i = 0; i < total_ele; i++) {
                    temp.push_back(static_cast<T>(tensor.data[i]));
                }
                std::sort(temp.begin(), temp.end());
                if (total_ele % 2 == 0) {
                    result_data[0] = (temp[total_ele/2 - 1] + temp[total_ele/2]) / 2;
                } else {
                    result_data[0] = temp[total_ele/2];
                }
                break;
            }
        }

        if (keepdims) {
            result_shape = vector_i32(tensor.ndim, 1);
        } else {
            result_shape = {1};
        }
        return {result_data, result_shape};
    }

    if (dim < 0 || dim >= tensor.ndim) {
        throw std::out_of_range("Reduction dimension out of range");
    }

    if (keepdims) {
        result_shape = tensor.shape;
        result_shape[dim] = 1;
    } else {
        result_shape.reserve(tensor.ndim - 1);
        for (int32_t i = 0; i < tensor.ndim; i++) {
            if (i != dim) {
                result_shape.push_back(tensor.shape[i]);
            }
        }
        if (result_shape.empty()) {
            result_shape.push_back(1);
        }
    }

    const int32_t result_size = calculate_size(result_shape, result_shape.size());
    const int32_t reduce_dim_size = tensor.shape[dim];
    
    using T = typename ResultType::value_type;
    result_data.resize(result_size);

    if (op != ReductionOp::MEDIAN) {
        for (int32_t i = 0; i < result_size; i++) {
            switch (op) {
                case ReductionOp::SUM:
                case ReductionOp::MEAN:
                    result_data[i] = 0;
                    break;
                case ReductionOp::MIN:
                    result_data[i] = std::numeric_limits<T>::infinity();
                    break;
                case ReductionOp::MAX:
                    result_data[i] = -std::numeric_limits<T>::infinity();
                    break;
                default:
                    break;
            }
        }

        vector_i32 input_strides = calculate_stride(tensor.shape, tensor.ndim);
        vector_i32 output_strides = calculate_stride(result_shape, result_shape.size());
        
        for (int32_t idx = 0; idx < total_ele; idx++) {
            int32_t input_idx = idx;
            int32_t output_idx = 0;
            
            for (int32_t d = 0, rd = 0; d < tensor.ndim; d++) {
                if (d == dim) continue;
                int32_t coord = (input_idx / input_strides[d]) % tensor.shape[d];
                output_idx += coord * output_strides[rd++];
            }

            T value = static_cast<T>(tensor.data[input_idx]);
            switch (op) {
                case ReductionOp::SUM:
                case ReductionOp::MEAN:
                    result_data[output_idx] += value;
                    break;
                case ReductionOp::MIN:
                    result_data[output_idx] = std::min(result_data[output_idx], value);
                    break;
                case ReductionOp::MAX:
                    result_data[output_idx] = std::max(result_data[output_idx], value);
                    break;
                default:
                    break;
            }
        }

        if (op == ReductionOp::MEAN) {
            for (int32_t i = 0; i < result_size; i++) {
                result_data[i] /= reduce_dim_size;
            }
        }
    } else {
        std::vector<std::vector<T>> slice_vals(result_size);
        for (auto& slice : slice_vals) {
            slice.reserve(reduce_dim_size);
        }

        vector_i32 input_strides = calculate_stride(tensor.shape, tensor.ndim);
        vector_i32 output_strides = calculate_stride(result_shape, result_shape.size());

        for (int32_t idx = 0; idx < total_ele; idx++) {
            int32_t input_idx = idx;
            int32_t output_idx = 0;
            
            for (int32_t d = 0, rd = 0; d < tensor.ndim; d++) {
                if (d == dim) continue;
                int32_t coord = (input_idx / input_strides[d]) % tensor.shape[d];
                output_idx += coord * output_strides[rd++];
            }

            slice_vals[output_idx].push_back(static_cast<T>(tensor.data[input_idx]));
        }

        for (int32_t i = 0; i < result_size; i++) {
            auto& vals = slice_vals[i];
            std::sort(vals.begin(), vals.end());
            if (vals.size() % 2 == 0) {
                result_data[i] = (vals[vals.size()/2 - 1] + vals[vals.size()/2]) / 2;
            } else {
                result_data[i] = vals[vals.size()/2];
            }
        }
    }

    return {result_data, result_shape};
}

template <typename T, typename U, typename Op>
std::pair<U, vector_i32> BaseConfigOp(T tensor1, T tensor2, Op op){

    U result_data;
    vector_i32 result_shape;
    int32_t max_dim = tensor1.ndim > tensor2.ndim ? tensor1.ndim: tensor2.ndim;

    vector_i32 result_stride1 = broadcast_stride(tensor1.shape, tensor1.stride, tensor1.ndim, max_dim);
    vector_i32 result_stride2 = broadcast_stride(tensor2.shape, tensor2.stride, tensor2.ndim, max_dim);

    int32_t allow = broadcast_shape(tensor1.shape, tensor2.shape, result_shape, tensor1.ndim, tensor2.ndim, max_dim);

    if (allow == -1){
        assert("The Shape is not broadcasted");
    }

    int32_t total_ele = calculate_size(result_shape, result_shape.size());

    vector_i32 result_stride = calculate_stride(result_shape, result_shape.size());

    result_data.resize(total_ele); 

    for(int32_t idx = 0; idx < total_ele; idx++){
        int32_t offset1 = 0; int32_t offset2 = 0;
        int n_idx = idx;

        update_offset(&offset1, &offset2, &n_idx, max_dim, result_stride, result_stride1, result_stride2);
        result_data[idx] = op(tensor1.data[offset1],tensor2.data[offset2]);
    }

    //I think this is the best way to delete the vector
    vector_i32().swap(result_stride1);
    vector_i32().swap(result_stride2);
    vector_i32().swap(result_stride);

    return {result_data, result_shape};
}

//arithmetic ops

//addition
std::pair<vector_f32, vector_i32> AddFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, std::function<float_t(float_t, float_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](float_t num1, float_t num2) {return num1 + num2;});
}

std::pair<vector_f64, vector_i32> AddFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, std::function<double_t(double_t, double_t)>>(tensor1, 
                                                                                                   tensor2, 
                                                                                                   [](double_t num1, double_t num2) {return num1 + num2;});   
}

std::pair<vector_i32, vector_i32> AddInt32(Int32TensorBase tensor1, Int32TensorBase tensor2){
    return BaseConfigOp<Int32TensorBase, vector_i32, std::function<int32_t(int32_t, int32_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](int32_t num1, int32_t num2) {return num1 + num2;});
}

std::pair<vector_i64, vector_i32> AddInt64(Int64TensorBase tensor1, Int64TensorBase tensor2){
    return BaseConfigOp<Int64TensorBase, vector_i64, std::function<int64_t(int64_t, int64_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](int64_t num1, int64_t num2) {return num1 + num2;});
}

std::pair<vector_bool, vector_i32> AddBool(BoolTensorBase tensor1, BoolTensorBase tensor2){
    return BaseConfigOp<BoolTensorBase, vector_bool, std::function<bool(bool, bool)>>(tensor1, 
                                                                                       tensor2, 
                                                                                       [](bool num1, bool num2) {return num1 + num2;});
}


//multiplication
std::pair<vector_f32, vector_i32> MulFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, std::function<float_t(float_t, float_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](float_t num1, float_t num2) {return num1 * num2;});
}

std::pair<vector_f64, vector_i32> MulFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, std::function<double_t(double_t, double_t)>>(tensor1, 
                                                                                                    tensor2, 
                                                                                                    [](double_t num1, double_t num2) {return num1 * num2;});   
}
    
std::pair<vector_i32, vector_i32> MulInt32(Int32TensorBase tensor1, Int32TensorBase tensor2){
    return BaseConfigOp<Int32TensorBase, vector_i32, std::function<int32_t(int32_t, int32_t)>>(tensor1, 
                                                                                                tensor2, 
                                                                                                [](int32_t num1, int32_t num2) {return num1 * num2;});
}

std::pair<vector_i64, vector_i32> MulInt64(Int64TensorBase tensor1, Int64TensorBase tensor2){
    return BaseConfigOp<Int64TensorBase, vector_i64, std::function<int64_t(int64_t, int64_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](int64_t num1, int64_t num2) {return num1 * num2;});
}

std::pair<vector_bool, vector_i32> MulBool(BoolTensorBase tensor1, BoolTensorBase tensor2){
    return BaseConfigOp<BoolTensorBase, vector_bool, std::function<bool(bool, bool)>>(tensor1, 
                                                                                      tensor2,
                                                                                      [](bool num1, bool num2) {return num1 * num2;});
}


std::pair<vector_f32, vector_i32> SubFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, std::function<float_t(float_t, float_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](float_t num1, float_t num2) {return num1 - num2;});
}

std::pair<vector_f64, vector_i32> SubFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, std::function<double_t(double_t, double_t)>>(tensor1, 
                                                                                                   tensor2, 
                                                                                                   [](double_t num1, double_t num2) {return num1 - num2;});   
}

std::pair<vector_i32, vector_i32> SubInt32(Int32TensorBase tensor1, Int32TensorBase tensor2){
    return BaseConfigOp<Int32TensorBase, vector_i32, std::function<int32_t(int32_t, int32_t)>>(tensor1, 
                                                                                                tensor2, 
                                                                                                [](int32_t num1, int32_t num2) {return num1 - num2;});
}

std::pair<vector_i64, vector_i32> SubInt64(Int64TensorBase tensor1, Int64TensorBase tensor2){
    return BaseConfigOp<Int64TensorBase, vector_i64, std::function<int64_t(int64_t, int64_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](int64_t num1, int64_t num2) {return num1 - num2;});
}

std::pair<vector_bool, vector_i32> SubBool(BoolTensorBase tensor1, BoolTensorBase tensor2){
    return BaseConfigOp<BoolTensorBase, vector_bool, std::function<bool(bool, bool)>>(tensor1, 
                                                                                      tensor2,
                                                                                      [](bool num1, bool num2) {return num1 - num2;});
}


std::pair<vector_f32, vector_i32> DivFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, std::function<float_t(float_t, float_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](float_t num1, float_t num2) {return num1 / num2;});
}

std::pair<vector_f64, vector_i32> DivFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, std::function<double_t(double_t, double_t)>>(tensor1, 
                                                                                                   tensor2, 
                                                                                                   [](double_t num1, double_t num2) {return num1 / num2;});   
}

std::pair<vector_i32, vector_i32> DivInt32(Int32TensorBase tensor1, Int32TensorBase tensor2){
    return BaseConfigOp<Int32TensorBase, vector_i32, std::function<int32_t(int32_t, int32_t)>>(tensor1, 
                                                                                                tensor2, 
                                                                                                [](int32_t num1, int32_t num2) {return num1 / num2;});
}

std::pair<vector_i64, vector_i32> DivInt64(Int64TensorBase tensor1, Int64TensorBase tensor2){
    return BaseConfigOp<Int64TensorBase, vector_i64, std::function<int64_t(int64_t, int64_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](int64_t num1, int64_t num2) {return num1 / num2;});
}

std::pair<vector_bool, vector_i32> DivBool(BoolTensorBase tensor1, BoolTensorBase tensor2){
    return BaseConfigOp<BoolTensorBase, vector_bool, std::function<bool(bool, bool)>>(tensor1, 
                                                                                      tensor2,
                                                                                      [](bool num1, bool num2) {return num1 / num2;});
}


std::pair<vector_f32, vector_i32> PowFloat32(FloatTensorBase tensor1, FloatTensorBase tensor2){
    return BaseConfigOp<FloatTensorBase, vector_f32, std::function<float_t(float_t, float_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](float_t num1, float_t num2) {return std::pow(num1, num2);});
}

std::pair<vector_f64, vector_i32> PowFloat64(DoubleTensorBase tensor1, DoubleTensorBase tensor2){
    return BaseConfigOp<DoubleTensorBase, vector_f64, std::function<double_t(double_t, double_t)>>(tensor1, 
                                                                                                   tensor2, 
                                                                                                   [](double_t num1, double_t num2) {return std::pow(num1, num2);});   
}

std::pair<vector_i32, vector_i32> PowInt32(Int32TensorBase tensor1, Int32TensorBase tensor2){
    return BaseConfigOp<Int32TensorBase, vector_i32, std::function<int32_t(int32_t, int32_t)>>(tensor1, 
                                                                                                tensor2, 
                                                                                                [](int32_t num1, int32_t num2) {return std::pow(num1, num2);});
}

std::pair<vector_i64, vector_i32> PowInt64(Int64TensorBase tensor1, Int64TensorBase tensor2){
    return BaseConfigOp<Int64TensorBase, vector_i64, std::function<int64_t(int64_t, int64_t)>>(tensor1, 
                                                                                               tensor2, 
                                                                                               [](int64_t num1, int64_t num2) {return std::pow(num1, num2);});
}


// Reduction ops
std::pair<vector_f32, vector_i32> SumFloat32(FloatTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<FloatTensorBase, vector_f32>(tensor, ReductionOp::SUM, dim, keepdims);
}

std::pair<vector_f64, vector_i32> SumFloat64(DoubleTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<DoubleTensorBase, vector_f64>(tensor, ReductionOp::SUM, dim, keepdims);
}

std::pair<vector_f32, vector_i32> SumInt32(Int32TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int32TensorBase, vector_f32>(tensor, ReductionOp::SUM, dim, keepdims);
}

std::pair<vector_f64, vector_i32> SumInt64(Int64TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int64TensorBase, vector_f64>(tensor, ReductionOp::SUM, dim, keepdims);
}


std::pair<vector_f32, vector_i32> MeanFloat32(FloatTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<FloatTensorBase, vector_f32>(tensor, ReductionOp::MEAN, dim, keepdims);
}

std::pair<vector_f64, vector_i32> MeanFloat64(DoubleTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<DoubleTensorBase, vector_f64>(tensor, ReductionOp::MEAN, dim, keepdims);
}

std::pair<vector_f32, vector_i32> MeanInt32(Int32TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int32TensorBase, vector_f32>(tensor, ReductionOp::MEAN, dim, keepdims);
}

std::pair<vector_f64, vector_i32> MeanInt64(Int64TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int64TensorBase, vector_f64>(tensor, ReductionOp::MEAN, dim, keepdims);
}

std::pair<vector_f32, vector_i32> MedianFloat32(FloatTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<FloatTensorBase, vector_f32>(tensor, ReductionOp::MEDIAN, dim, keepdims);
}

std::pair<vector_f64, vector_i32> MedianFloat64(DoubleTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<DoubleTensorBase, vector_f64>(tensor, ReductionOp::MEDIAN, dim, keepdims);
}

std::pair<vector_f32, vector_i32> MedianInt32(Int32TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int32TensorBase, vector_f32>(tensor, ReductionOp::MEDIAN, dim, keepdims);
}  

std::pair<vector_f64, vector_i32> MedianInt64(Int64TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int64TensorBase, vector_f64>(tensor, ReductionOp::MEDIAN, dim, keepdims);
}

std::pair<vector_f32, vector_i32> MinFloat32(FloatTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<FloatTensorBase, vector_f32>(tensor, ReductionOp::MIN, dim, keepdims);
}

std::pair<vector_f64, vector_i32> MinFloat64(DoubleTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<DoubleTensorBase, vector_f64>(tensor, ReductionOp::MIN, dim, keepdims);
}

std::pair<vector_f32, vector_i32> MinInt32(Int32TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int32TensorBase, vector_f32>(tensor, ReductionOp::MIN, dim, keepdims);
}

std::pair<vector_f64, vector_i32> MinInt64(Int64TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int64TensorBase, vector_f64>(tensor, ReductionOp::MIN, dim, keepdims);
}

std::pair<vector_f32, vector_i32> MaxFloat32(FloatTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<FloatTensorBase, vector_f32>(tensor, ReductionOp::MAX, dim, keepdims);
}

std::pair<vector_f64, vector_i32> MaxFloat64(DoubleTensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<DoubleTensorBase, vector_f64>(tensor, ReductionOp::MAX, dim, keepdims);
}

std::pair<vector_f32, vector_i32> MaxInt32(Int32TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int32TensorBase, vector_f32>(tensor, ReductionOp::MAX, dim, keepdims);
}

std::pair<vector_f64, vector_i32> MaxInt64(Int64TensorBase tensor, int32_t dim, bool keepdims) {
    return ReduceConfig<Int64TensorBase, vector_f64>(tensor, ReductionOp::MAX, dim, keepdims);
}