#pragma once

#include "tensor.h"
#include <memory>
#include <vector>
#include <cmath>

// Base interface for all positional encoding types
class PositionalEncodingBase {
public:
    virtual ~PositionalEncodingBase() = default;
    
    // Forward pass - adds positional information to input
    virtual Tensor forward(const Tensor& x) = 0;
    
    // Get parameters (for learned encodings)
    virtual std::vector<Tensor*> parameters() { return {}; }
    
    // Initialize parameters (for learned encodings)
    virtual void init_parameters() {}
    
    // Get encoding type name
    virtual std::string type_name() const = 0;
    
    // Get maximum sequence length supported
    virtual size_t max_seq_len() const = 0;
    
    // Get model dimension
    virtual size_t d_model() const = 0;
};

// Sinusoidal positional encoding (original Transformer paper)
class SinusoidalPositionalEncoding : public PositionalEncodingBase {
private:
    size_t d_model_;
    size_t max_seq_len_;
    std::unique_ptr<Tensor> encoding_;
    bool cache_computed_;

public:
    SinusoidalPositionalEncoding(size_t d_model, size_t max_seq_len = 5000);
    
    Tensor forward(const Tensor& x) override;
    std::string type_name() const override { return "sinusoidal"; }
    size_t max_seq_len() const override { return max_seq_len_; }
    size_t d_model() const override { return d_model_; }
    
private:
    void compute_encoding();
    float get_position_angle(size_t pos, size_t i) const;
};

// Learned positional encoding (trainable parameters)
class LearnedPositionalEncoding : public PositionalEncodingBase {
private:
    size_t d_model_;
    size_t max_seq_len_;
    std::unique_ptr<Tensor> encoding_;

public:
    LearnedPositionalEncoding(size_t d_model, size_t max_seq_len = 5000);
    
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor*> parameters() override;
    void init_parameters() override;
    std::string type_name() const override { return "learned"; }
    size_t max_seq_len() const override { return max_seq_len_; }
    size_t d_model() const override { return d_model_; }
};

// Rotary Position Embedding (RoPE) - applies rotation to Q and K in attention
class RotaryPositionalEncoding : public PositionalEncodingBase {
private:
    size_t d_model_;
    size_t max_seq_len_;
    float base_;
    std::unique_ptr<Tensor> cos_cache_;
    std::unique_ptr<Tensor> sin_cache_;
    bool cache_computed_;

public:
    RotaryPositionalEncoding(size_t d_model, size_t max_seq_len = 5000, float base = 10000.0f);
    
    // RoPE doesn't add to input like other encodings, it rotates Q and K
    Tensor forward(const Tensor& x) override;
    
    // Main RoPE functions for applying rotation to queries and keys
    Tensor apply_rotary_pos_emb(const Tensor& x, size_t seq_len, size_t offset = 0);
    std::pair<Tensor, Tensor> apply_rotary_pos_emb_qk(const Tensor& q, const Tensor& k, 
                                                      size_t seq_len, size_t offset = 0);
    
    std::string type_name() const override { return "rotary"; }
    size_t max_seq_len() const override { return max_seq_len_; }
    size_t d_model() const override { return d_model_; }
    
private:
    void compute_freqs();
    Tensor rotate_half(const Tensor& x);
    std::pair<Tensor, Tensor> get_cos_sin(size_t seq_len, size_t offset = 0);
};

// Factory for creating positional encodings
enum class PositionalEncodingType {
    SINUSOIDAL,
    LEARNED,
    ROTARY
};

class PositionalEncodingFactory {
public:
    static std::unique_ptr<PositionalEncodingBase> create(
        PositionalEncodingType type,
        size_t d_model,
        size_t max_seq_len = 5000,
        float rope_base = 10000.0f
    );
    
    static PositionalEncodingType string_to_type(const std::string& type_str);
    static std::string type_to_string(PositionalEncodingType type);
};

// Configuration struct for positional encoding
struct PositionalEncodingConfig {
    PositionalEncodingType type = PositionalEncodingType::SINUSOIDAL;
    size_t d_model = 512;
    size_t max_seq_len = 5000;
    float rope_base = 10000.0f;
    
    void validate() const;
};

// Helper functions for RoPE integration with attention
namespace RoPEUtils {
    // Apply RoPE to multi-head attention queries and keys
    std::pair<Tensor, Tensor> apply_rope_to_qk(
        const Tensor& q, const Tensor& k,
        RotaryPositionalEncoding& rope,
        size_t seq_len, size_t offset = 0
    );
    
    // Check if tensor dimensions are compatible with RoPE
    bool is_rope_compatible(const Tensor& tensor, size_t expected_d_model);
}