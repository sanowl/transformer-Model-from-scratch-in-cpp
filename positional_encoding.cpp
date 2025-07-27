#include "positional_encoding.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>

// ============================================================================
// SinusoidalPositionalEncoding Implementation
// ============================================================================

SinusoidalPositionalEncoding::SinusoidalPositionalEncoding(size_t d_model, size_t max_seq_len)
    : d_model_(d_model), max_seq_len_(max_seq_len), cache_computed_(false) {
    
    if (d_model == 0) {
        throw std::invalid_argument("d_model must be positive");
    }
    if (max_seq_len == 0) {
        throw std::invalid_argument("max_seq_len must be positive");
    }
    
    // Create encoding tensor [max_seq_len, d_model]
    encoding_ = std::make_unique<Tensor>(std::vector<size_t>{max_seq_len, d_model});
}

Tensor SinusoidalPositionalEncoding::forward(const Tensor& x) {
    if (!cache_computed_) {
        compute_encoding();
        cache_computed_ = true;
    }
    
    // Validate input dimensions - expect [batch_size, seq_len, d_model]
    if (x.ndim() != 3) {
        throw std::invalid_argument("Input tensor must have 3 dimensions [batch_size, seq_len, d_model]");
    }
    if (x.shape()[2] != d_model_) {
        throw std::invalid_argument("Input tensor last dimension must match d_model");
    }
    
    size_t batch_size = x.shape()[0];
    size_t seq_len = x.shape()[1];
    if (seq_len > max_seq_len_) {
        throw std::invalid_argument("Sequence length exceeds maximum supported length");
    }
    
    // Create result tensor with same shape as input
    Tensor result(x.shape());
    
    // Add positional encoding to each batch element
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < d_model_; ++d) {
                result[{b, s, d}] = x[{b, s, d}] + (*encoding_)[{s, d}];
            }
        }
    }
    
    return result;
}

void SinusoidalPositionalEncoding::compute_encoding() {
    for (size_t pos = 0; pos < max_seq_len_; ++pos) {
        for (size_t i = 0; i < d_model_; ++i) {
            float angle = get_position_angle(pos, i);
            if (i % 2 == 0) {
                (*encoding_)[{pos, i}] = std::sin(angle);
            } else {
                (*encoding_)[{pos, i}] = std::cos(angle);
            }
        }
    }
}

float SinusoidalPositionalEncoding::get_position_angle(size_t pos, size_t i) const {
    float div_term = std::exp((i / 2) * 2 * (-std::log(10000.0f) / d_model_));
    return pos * div_term;
}

// ============================================================================
// LearnedPositionalEncoding Implementation
// ============================================================================

LearnedPositionalEncoding::LearnedPositionalEncoding(size_t d_model, size_t max_seq_len)
    : d_model_(d_model), max_seq_len_(max_seq_len) {
    
    if (d_model == 0) {
        throw std::invalid_argument("d_model must be positive");
    }
    if (max_seq_len == 0) {
        throw std::invalid_argument("max_seq_len must be positive");
    }
    
    // Create learnable encoding tensor [max_seq_len, d_model]
    encoding_ = std::make_unique<Tensor>(std::vector<size_t>{max_seq_len, d_model}, true);
    init_parameters();
}

Tensor LearnedPositionalEncoding::forward(const Tensor& x) {
    // Validate input dimensions - expect [batch_size, seq_len, d_model]
    if (x.ndim() != 3) {
        throw std::invalid_argument("Input tensor must have 3 dimensions [batch_size, seq_len, d_model]");
    }
    if (x.shape()[2] != d_model_) {
        throw std::invalid_argument("Input tensor last dimension must match d_model");
    }
    
    size_t batch_size = x.shape()[0];
    size_t seq_len = x.shape()[1];
    if (seq_len > max_seq_len_) {
        throw std::invalid_argument("Sequence length exceeds maximum supported length");
    }
    
    // Create result tensor with same shape as input
    Tensor result(x.shape());
    
    // Add learned positional encoding to each batch element
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < d_model_; ++d) {
                result[{b, s, d}] = x[{b, s, d}] + (*encoding_)[{s, d}];
            }
        }
    }
    
    return result;
}

std::vector<Tensor*> LearnedPositionalEncoding::parameters() {
    return {encoding_.get()};
}

void LearnedPositionalEncoding::init_parameters() {
    // Initialize with small random values using normal distribution
    encoding_->normal_(0.0f, 0.02f);
}

// ============================================================================
// RotaryPositionalEncoding Implementation
// ============================================================================

RotaryPositionalEncoding::RotaryPositionalEncoding(size_t d_model, size_t max_seq_len, float base)
    : d_model_(d_model), max_seq_len_(max_seq_len), base_(base), cache_computed_(false) {
    
    if (d_model == 0) {
        throw std::invalid_argument("d_model must be positive");
    }
    if (d_model % 2 != 0) {
        throw std::invalid_argument("d_model must be even for RoPE");
    }
    if (max_seq_len == 0) {
        throw std::invalid_argument("max_seq_len must be positive");
    }
    if (base <= 0) {
        throw std::invalid_argument("base must be positive");
    }
    
    // Create frequency cache [max_seq_len, d_model/2]
    cos_cache_ = std::make_unique<Tensor>(std::vector<size_t>{max_seq_len, d_model / 2});
    sin_cache_ = std::make_unique<Tensor>(std::vector<size_t>{max_seq_len, d_model / 2});
}

Tensor RotaryPositionalEncoding::forward(const Tensor& x) {
    // RoPE doesn't directly add to input like other encodings
    // This method exists for interface compatibility
    // Actual rotation should be done via apply_rotary_pos_emb
    return x;
}

Tensor RotaryPositionalEncoding::apply_rotary_pos_emb(const Tensor& x, size_t seq_len, size_t offset) {
    if (!cache_computed_) {
        compute_freqs();
        cache_computed_ = true;
    }
    
    if (x.ndim() < 3) {
        throw std::invalid_argument("Input tensor must have at least 3 dimensions [batch, seq_len, d_model]");
    }
    if (x.shape()[x.ndim() - 1] != d_model_) {
        throw std::invalid_argument("Input tensor last dimension must match d_model");
    }
    if (seq_len + offset > max_seq_len_) {
        throw std::invalid_argument("Sequence length + offset exceeds maximum supported length");
    }
    
    auto [cos_emb, sin_emb] = get_cos_sin(seq_len, offset);
    
    // Get tensor dimensions
    std::vector<size_t> input_shape = x.shape();
    size_t batch_size = input_shape[0];
    size_t actual_seq_len = input_shape[1];
    size_t d_model = input_shape[2];
    
    if (actual_seq_len != seq_len) {
        throw std::invalid_argument("Sequence length mismatch");
    }
    
    // Create result tensor with same shape
    Tensor result(input_shape);
    
    // Apply RoPE rotation using optimized computation
    size_t half_dim = d_model / 2;
    
    // Process each batch and sequence position
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            // Get cos/sin values for this position
            for (size_t d = 0; d < half_dim; ++d) {
                float cos_val = cos_emb[{s, d}];
                float sin_val = sin_emb[{s, d}];
                
                // Get the two elements to rotate
                float x1 = x[{b, s, d}];
                float x2 = x[{b, s, d + half_dim}];
                
                // Apply rotation matrix: [cos -sin; sin cos] * [x1; x2]
                result[{b, s, d}] = x1 * cos_val - x2 * sin_val;
                result[{b, s, d + half_dim}] = x1 * sin_val + x2 * cos_val;
            }
        }
    }
    
    return result;
}

std::pair<Tensor, Tensor> RotaryPositionalEncoding::apply_rotary_pos_emb_qk(
    const Tensor& q, const Tensor& k, size_t seq_len, size_t offset) {
    
    Tensor q_rotated = apply_rotary_pos_emb(q, seq_len, offset);
    Tensor k_rotated = apply_rotary_pos_emb(k, seq_len, offset);
    
    return {q_rotated, k_rotated};
}

void RotaryPositionalEncoding::compute_freqs() {
    size_t dim = d_model_ / 2;
    
    // Compute frequency for each dimension pair
    for (size_t i = 0; i < dim; ++i) {
        float freq = 1.0f / std::pow(base_, static_cast<float>(i * 2) / d_model_);
        
        // Compute cos and sin for each position
        for (size_t pos = 0; pos < max_seq_len_; ++pos) {
            float angle = pos * freq;
            (*cos_cache_)[{pos, i}] = std::cos(angle);
            (*sin_cache_)[{pos, i}] = std::sin(angle);
        }
    }
}

Tensor RotaryPositionalEncoding::rotate_half(const Tensor& x) {
    std::vector<size_t> input_shape = x.shape();
    size_t half_dim = d_model_ / 2;
    Tensor result(input_shape);
    
    // Handle 3D tensor: [batch_size, seq_len, d_model]
    if (input_shape.size() == 3) {
        size_t batch_size = input_shape[0];
        size_t seq_len = input_shape[1];
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t d = 0; d < half_dim; ++d) {
                    // Rotate: swap halves and negate first half
                    result[{b, s, d}] = -x[{b, s, d + half_dim}];
                    result[{b, s, d + half_dim}] = x[{b, s, d}];
                }
            }
        }
    }
    // Handle 2D tensor: [seq_len, d_model]
    else if (input_shape.size() == 2) {
        size_t seq_len = input_shape[0];
        
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < half_dim; ++d) {
                result[{s, d}] = -x[{s, d + half_dim}];
                result[{s, d + half_dim}] = x[{s, d}];
            }
        }
    }
    else {
        throw std::invalid_argument("rotate_half expects 2D or 3D tensor");
    }
    
    return result;
}

std::pair<Tensor, Tensor> RotaryPositionalEncoding::get_cos_sin(size_t seq_len, size_t offset) {
    Tensor cos_emb({seq_len, d_model_ / 2});
    Tensor sin_emb({seq_len, d_model_ / 2});
    
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_model_ / 2; ++j) {
            cos_emb[{i, j}] = (*cos_cache_)[{i + offset, j}];
            sin_emb[{i, j}] = (*sin_cache_)[{i + offset, j}];
        }
    }
    
    return {cos_emb, sin_emb};
}

// ============================================================================
// PositionalEncodingFactory Implementation
// ============================================================================

std::unique_ptr<PositionalEncodingBase> PositionalEncodingFactory::create(
    PositionalEncodingType type, size_t d_model, size_t max_seq_len, float rope_base) {
    
    switch (type) {
        case PositionalEncodingType::SINUSOIDAL:
            return std::make_unique<SinusoidalPositionalEncoding>(d_model, max_seq_len);
        
        case PositionalEncodingType::LEARNED:
            return std::make_unique<LearnedPositionalEncoding>(d_model, max_seq_len);
        
        case PositionalEncodingType::ROTARY:
            return std::make_unique<RotaryPositionalEncoding>(d_model, max_seq_len, rope_base);
        
        default:
            throw std::invalid_argument("Unknown positional encoding type");
    }
}

PositionalEncodingType PositionalEncodingFactory::string_to_type(const std::string& type_str) {
    if (type_str == "sinusoidal") return PositionalEncodingType::SINUSOIDAL;
    if (type_str == "learned") return PositionalEncodingType::LEARNED;
    if (type_str == "rotary" || type_str == "rope") return PositionalEncodingType::ROTARY;
    
    throw std::invalid_argument("Unknown positional encoding type string: " + type_str);
}

std::string PositionalEncodingFactory::type_to_string(PositionalEncodingType type) {
    switch (type) {
        case PositionalEncodingType::SINUSOIDAL: return "sinusoidal";
        case PositionalEncodingType::LEARNED: return "learned";
        case PositionalEncodingType::ROTARY: return "rotary";
        default: return "unknown";
    }
}

// ============================================================================
// PositionalEncodingConfig Implementation
// ============================================================================

void PositionalEncodingConfig::validate() const {
    if (d_model == 0) {
        throw std::invalid_argument("d_model must be positive");
    }
    if (max_seq_len == 0) {
        throw std::invalid_argument("max_seq_len must be positive");
    }
    if (type == PositionalEncodingType::ROTARY && d_model % 2 != 0) {
        throw std::invalid_argument("d_model must be even for RoPE");
    }
    if (rope_base <= 0) {
        throw std::invalid_argument("rope_base must be positive");
    }
}

// ============================================================================
// RoPEUtils Implementation
// ============================================================================

namespace RoPEUtils {
    std::pair<Tensor, Tensor> apply_rope_to_qk(
        const Tensor& q, const Tensor& k,
        RotaryPositionalEncoding& rope,
        size_t seq_len, size_t offset) {
        
        return rope.apply_rotary_pos_emb_qk(q, k, seq_len, offset);
    }
    
    bool is_rope_compatible(const Tensor& tensor, size_t expected_d_model) {
        return tensor.ndim() >= 2 && 
               tensor.shape()[tensor.ndim() - 1] == expected_d_model &&
               expected_d_model % 2 == 0;
    }
}