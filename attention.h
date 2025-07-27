#pragma once

#include "tensor.h"
#include <memory>
#include <vector>

// Forward declarations
class PositionalEncodingBase;
class RotaryPositionalEncoding;

class MultiHeadAttention {
private:
    size_t d_model_;
    size_t n_heads_;
    size_t d_k_;
    size_t d_v_;
    float dropout_rate_;
    
    std::unique_ptr<Tensor> W_q_;
    std::unique_ptr<Tensor> W_k_;
    std::unique_ptr<Tensor> W_v_;
    std::unique_ptr<Tensor> W_o_;
    
    std::unique_ptr<Tensor> b_q_;
    std::unique_ptr<Tensor> b_k_;
    std::unique_ptr<Tensor> b_v_;
    std::unique_ptr<Tensor> b_o_;
    
    // RoPE support
    RotaryPositionalEncoding* rope_encoding_;
    bool use_rope_;

public:
    MultiHeadAttention(size_t d_model, size_t n_heads, float dropout = 0.1f);
    ~MultiHeadAttention() = default;
    
    MultiHeadAttention(const MultiHeadAttention&) = delete;
    MultiHeadAttention& operator=(const MultiHeadAttention&) = delete;
    MultiHeadAttention(MultiHeadAttention&&) = default;
    MultiHeadAttention& operator=(MultiHeadAttention&&) = default;
    
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value, 
                   const Tensor* mask = nullptr, bool training = true);
    
    Tensor scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                       const Tensor* mask = nullptr, bool training = true);
    
    void init_parameters();
    
    std::vector<Tensor*> parameters();
    
    // RoPE integration methods
    void set_rope_encoding(RotaryPositionalEncoding* rope_encoding);
    void enable_rope(bool enable) { use_rope_ = enable; }
    bool is_rope_enabled() const { return use_rope_; }
    
private:
    Tensor linear_transform(const Tensor& input, const Tensor& weight, const Tensor& bias);
    Tensor split_heads(const Tensor& x, size_t batch_size, size_t seq_len);
    Tensor combine_heads(const Tensor& x, size_t batch_size, size_t seq_len);
    Tensor apply_mask(const Tensor& scores, const Tensor& mask);
    Tensor dropout(const Tensor& x, bool training);
};

class PositionalEncoding {
private:
    size_t d_model_;
    size_t max_seq_len_;
    std::unique_ptr<Tensor> encoding_;

public:
    PositionalEncoding(size_t d_model, size_t max_seq_len = 5000);
    
    Tensor forward(const Tensor& x);
    
private:
    void compute_encoding();
};

class LayerNorm {
private:
    size_t d_model_;
    float eps_;
    std::unique_ptr<Tensor> gamma_;
    std::unique_ptr<Tensor> beta_;

public:
    LayerNorm(size_t d_model, float eps = 1e-6f);
    
    Tensor forward(const Tensor& x);
    void init_parameters();
    std::vector<Tensor*> parameters();
};

class FeedForward {
private:
    size_t d_model_;
    size_t d_ff_;
    float dropout_rate_;
    
    std::unique_ptr<Tensor> W1_;
    std::unique_ptr<Tensor> W2_;
    std::unique_ptr<Tensor> b1_;
    std::unique_ptr<Tensor> b2_;

public:
    FeedForward(size_t d_model, size_t d_ff, float dropout = 0.1f);
    
    Tensor forward(const Tensor& x, bool training = true);
    void init_parameters();
    std::vector<Tensor*> parameters();

private:
    Tensor dropout(const Tensor& x, bool training);
};