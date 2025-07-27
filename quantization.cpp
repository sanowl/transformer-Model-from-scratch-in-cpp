#include "quantization.h"
#include <algorithm>
#include <cmath>
#include <limits>

QuantizedTensor::QuantizedTensor(const Tensor& tensor, float scale, int8_t zero_point)
    : shape_(tensor.shape()), scale_(scale), zero_point_(zero_point) {
    
    size_t total_size = tensor.numel();
    quantized_data_.resize(total_size);
    
    for (size_t i = 0; i < tensor.matrix().rows(); ++i) {
        for (size_t j = 0; j < tensor.matrix().cols(); ++j) {
            size_t flat_idx = i * tensor.matrix().cols() + j;
            quantized_data_[flat_idx] = quantize_value(tensor.matrix()(i, j));
        }
    }
}

QuantizedTensor::QuantizedTensor(const std::vector<size_t>& shape, float scale, int8_t zero_point)
    : shape_(shape), scale_(scale), zero_point_(zero_point) {
    
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    quantized_data_.resize(total_size, zero_point);
}

int8_t QuantizedTensor::quantize_value(float value) const {
    float quantized_fp = std::round(value / scale_) + zero_point_;
    return static_cast<int8_t>(std::clamp(quantized_fp, -128.0f, 127.0f));
}

float QuantizedTensor::dequantize_value(int8_t value) const {
    return scale_ * (value - zero_point_);
}

Tensor QuantizedTensor::dequantize() const {
    Tensor result(shape_);
    
    for (size_t i = 0; i < result.matrix().rows(); ++i) {
        for (size_t j = 0; j < result.matrix().cols(); ++j) {
            size_t flat_idx = i * result.matrix().cols() + j;
            result.matrix()(i, j) = dequantize_value(quantized_data_[flat_idx]);
        }
    }
    
    return result;
}

void QuantizedTensor::quantize_from(const Tensor& tensor) {
    if (tensor.shape() != shape_) {
        throw std::invalid_argument("Shape mismatch in quantize_from");
    }
    
    for (size_t i = 0; i < tensor.matrix().rows(); ++i) {
        for (size_t j = 0; j < tensor.matrix().cols(); ++j) {
            size_t flat_idx = i * tensor.matrix().cols() + j;
            quantized_data_[flat_idx] = quantize_value(tensor.matrix()(i, j));
        }
    }
}

QuantizedTensor QuantizedTensor::quantized_matmul(const QuantizedTensor& other, 
                                                 float output_scale, int8_t output_zero_point) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("Quantized matmul requires 2D tensors");
    }
    
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Incompatible shapes for quantized matmul");
    }
    
    size_t m = shape_[0];
    size_t n = other.shape_[1];
    size_t k = shape_[1];
    
    std::vector<size_t> result_shape = {m, n};
    QuantizedTensor result(result_shape, output_scale, output_zero_point);
    
    // Perform int8 matrix multiplication with accumulation in int32
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t accumulator = 0;
            
            for (size_t l = 0; l < k; ++l) {
                int8_t a_val = quantized_data_[i * k + l];
                int8_t b_val = other.quantized_data_[l * n + j];
                
                accumulator += static_cast<int32_t>(a_val - zero_point_) * 
                              static_cast<int32_t>(b_val - other.zero_point_);
            }
            
            // Rescale and requantize
            float fp_result = accumulator * scale_ * other.scale_;
            int8_t quantized_result = static_cast<int8_t>(
                std::clamp(std::round(fp_result / output_scale) + output_zero_point, 
                          -128.0f, 127.0f));
            
            result.quantized_data_[i * n + j] = quantized_result;
        }
    }
    
    return result;
}

std::pair<float, int8_t> QuantizationUtils::compute_quantization_params(const Tensor& tensor) {
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (size_t i = 0; i < tensor.matrix().rows(); ++i) {
        for (size_t j = 0; j < tensor.matrix().cols(); ++j) {
            float val = tensor.matrix()(i, j);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }
    
    return compute_quantization_params(min_val, max_val);
}

std::pair<float, int8_t> QuantizationUtils::compute_quantization_params(float min_val, float max_val) {
    // Ensure min_val and max_val are not equal
    if (std::abs(max_val - min_val) < 1e-7f) {
        max_val = min_val + 1e-7f;
    }
    
    // Compute scale and zero point for asymmetric quantization
    float scale = (max_val - min_val) / (QMAX - QMIN);
    
    // Zero point should map to 0.0f in floating point
    float zero_point_fp = QMIN - min_val / scale;
    int8_t zero_point = static_cast<int8_t>(std::clamp(std::round(zero_point_fp), 
                                                      static_cast<float>(QMIN), 
                                                      static_cast<float>(QMAX)));
    
    return {scale, zero_point};
}

QuantizedTensor QuantizationUtils::quantize_tensor(const Tensor& tensor) {
    auto [scale, zero_point] = compute_quantization_params(tensor);
    return QuantizedTensor(tensor, scale, zero_point);
}

QuantizedTensor QuantizationUtils::quantize_tensor_symmetric(const Tensor& tensor) {
    float max_abs = 0.0f;
    
    for (size_t i = 0; i < tensor.matrix().rows(); ++i) {
        for (size_t j = 0; j < tensor.matrix().cols(); ++j) {
            max_abs = std::max(max_abs, std::abs(tensor.matrix()(i, j)));
        }
    }
    
    // Symmetric quantization: zero_point = 0, scale based on max absolute value
    float scale = max_abs / 127.0f;  // Use 127 instead of 128 for symmetry
    if (scale == 0.0f) scale = 1e-7f;  // Avoid division by zero
    
    return QuantizedTensor(tensor, scale, 0);
}

void QuantizationUtils::quantize_model_weights(std::vector<Tensor*>& parameters, 
                                              std::vector<QuantizedTensor>& quantized_weights) {
    quantized_weights.clear();
    quantized_weights.reserve(parameters.size());
    
    for (auto* param : parameters) {
        quantized_weights.push_back(quantize_tensor_symmetric(*param));
    }
}

Int8MultiHeadAttention::Int8MultiHeadAttention(const MultiHeadAttention& fp32_attention)
    : d_model_(512), n_heads_(8) {  // These should be extracted from fp32_attention
    
    d_k_ = d_model_ / n_heads_;
    
    // In a real implementation, you would extract the actual weights from fp32_attention
    // For now, we'll create dummy quantized weights
    Tensor dummy_weight(std::vector<size_t>{d_model_, d_model_});
    dummy_weight.normal_(0.0f, 0.02f);
    
    W_q_ = std::make_unique<QuantizedTensor>(QuantizationUtils::quantize_tensor_symmetric(dummy_weight));
    W_k_ = std::make_unique<QuantizedTensor>(QuantizationUtils::quantize_tensor_symmetric(dummy_weight));
    W_v_ = std::make_unique<QuantizedTensor>(QuantizationUtils::quantize_tensor_symmetric(dummy_weight));
    W_o_ = std::make_unique<QuantizedTensor>(QuantizationUtils::quantize_tensor_symmetric(dummy_weight));
}

Tensor Int8MultiHeadAttention::quantized_linear(const Tensor& input, const QuantizedTensor& weight) {
    // Quantize input
    QuantizedTensor q_input = QuantizationUtils::quantize_tensor_symmetric(input);
    
    // Compute output scale (simplified)
    float output_scale = q_input.scale() * weight.scale();
    
    // Perform quantized matrix multiplication
    QuantizedTensor q_output = q_input.quantized_matmul(weight, output_scale, 0);
    
    // Dequantize result
    return q_output.dequantize();
}

Tensor Int8MultiHeadAttention::int8_matmul_with_dequant(const Tensor& a, const QuantizedTensor& b) {
    QuantizedTensor q_a = QuantizationUtils::quantize_tensor_symmetric(a);
    float output_scale = q_a.scale() * b.scale();
    QuantizedTensor q_result = q_a.quantized_matmul(b, output_scale, 0);
    return q_result.dequantize();
}

Tensor Int8MultiHeadAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value,
                                      const Tensor* mask) {
    
    // Quantized linear transformations
    Tensor Q = quantized_linear(query, *W_q_);
    Tensor K = quantized_linear(key, *W_k_);
    Tensor V = quantized_linear(value, *W_v_);
    
    // Reshape for multi-head attention (simplified)
    size_t batch_size = 1;
    size_t seq_len = query.shape()[0];
    
    // Compute attention scores
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k_));
    Tensor K_transposed = K.transpose();
    Tensor scores = Q.matmul(K_transposed);
    scores *= scale;
    
    // Apply mask if provided
    if (mask) {
        const float neg_inf = -1e9f;
        for (size_t i = 0; i < scores.matrix().rows(); ++i) {
            for (size_t j = 0; j < scores.matrix().cols(); ++j) {
                size_t mask_i = i % mask->matrix().rows();
                size_t mask_j = j % mask->matrix().cols();
                
                if (mask->matrix()(mask_i, mask_j) == 0.0f) {
                    scores.matrix()(i, j) = neg_inf;
                }
            }
        }
    }
    
    // Apply softmax
    Tensor attention_weights = scores.softmax(-1);
    
    // Apply attention to values
    Tensor attention_output = attention_weights.matmul(V);
    
    // Final linear transformation
    Tensor output = quantized_linear(attention_output, *W_o_);
    
    return output;
}