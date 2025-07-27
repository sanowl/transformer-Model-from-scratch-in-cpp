#pragma once

#include "tensor.h"
#include "attention.h"
#include <cstdint>
#include <memory>

class QuantizedTensor {
private:
    std::vector<int8_t> quantized_data_;
    std::vector<size_t> shape_;
    float scale_;
    int8_t zero_point_;

public:
    QuantizedTensor(const Tensor& tensor, float scale, int8_t zero_point);
    QuantizedTensor(const std::vector<size_t>& shape, float scale, int8_t zero_point);
    
    Tensor dequantize() const;
    void quantize_from(const Tensor& tensor);
    
    const std::vector<size_t>& shape() const { return shape_; }
    float scale() const { return scale_; }
    int8_t zero_point() const { return zero_point_; }
    
    const int8_t* data() const { return quantized_data_.data(); }
    size_t size() const { return quantized_data_.size(); }
    
    QuantizedTensor quantized_matmul(const QuantizedTensor& other, float output_scale, int8_t output_zero_point) const;
    
private:
    int8_t quantize_value(float value) const;
    float dequantize_value(int8_t value) const;
};

class QuantizationUtils {
public:
    static std::pair<float, int8_t> compute_quantization_params(const Tensor& tensor);
    static std::pair<float, int8_t> compute_quantization_params(float min_val, float max_val);
    
    static QuantizedTensor quantize_tensor(const Tensor& tensor);
    static QuantizedTensor quantize_tensor_symmetric(const Tensor& tensor);
    
    static void quantize_model_weights(std::vector<Tensor*>& parameters, 
                                      std::vector<QuantizedTensor>& quantized_weights);
    
private:
    static constexpr int8_t QMIN = -128;
    static constexpr int8_t QMAX = 127;
};

class Int8MultiHeadAttention {
private:
    size_t d_model_;
    size_t n_heads_;
    size_t d_k_;
    
    std::unique_ptr<QuantizedTensor> W_q_;
    std::unique_ptr<QuantizedTensor> W_k_;
    std::unique_ptr<QuantizedTensor> W_v_;
    std::unique_ptr<QuantizedTensor> W_o_;

public:
    Int8MultiHeadAttention(const MultiHeadAttention& fp32_attention);
    
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value, 
                   const Tensor* mask = nullptr);
    
private:
    Tensor quantized_linear(const Tensor& input, const QuantizedTensor& weight);
    Tensor int8_matmul_with_dequant(const Tensor& a, const QuantizedTensor& b);
};