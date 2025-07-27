#include "attention.h"
#include "positional_encoding.h"
#include <stdexcept>

// Complete implementation of MultiHeadAttention with RoPE support

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t n_heads, float dropout)
    : d_model_(d_model), n_heads_(n_heads), dropout_rate_(dropout), 
      rope_encoding_(nullptr), use_rope_(false) {
    
    if (d_model == 0) {
        throw std::invalid_argument("d_model must be positive");
    }
    if (n_heads == 0) {
        throw std::invalid_argument("n_heads must be positive");
    }
    if (d_model % n_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by n_heads");
    }
    if (dropout < 0.0f || dropout > 1.0f) {
        throw std::invalid_argument("dropout rate must be between 0 and 1");
    }
    
    d_k_ = d_model / n_heads;
    d_v_ = d_model / n_heads;
    
    // Initialize weight matrices
    W_q_ = std::make_unique<Tensor>(std::vector<size_t>{d_model, d_model}, true);
    W_k_ = std::make_unique<Tensor>(std::vector<size_t>{d_model, d_model}, true);
    W_v_ = std::make_unique<Tensor>(std::vector<size_t>{d_model, d_model}, true);
    W_o_ = std::make_unique<Tensor>(std::vector<size_t>{d_model, d_model}, true);
    
    // Initialize bias vectors
    b_q_ = std::make_unique<Tensor>(std::vector<size_t>{d_model}, true);
    b_k_ = std::make_unique<Tensor>(std::vector<size_t>{d_model}, true);
    b_v_ = std::make_unique<Tensor>(std::vector<size_t>{d_model}, true);
    b_o_ = std::make_unique<Tensor>(std::vector<size_t>{d_model}, true);
    
    init_parameters();
}

void MultiHeadAttention::set_rope_encoding(RotaryPositionalEncoding* rope_encoding) {
    rope_encoding_ = rope_encoding;
    if (rope_encoding && rope_encoding->d_model() != d_model_) {
        throw std::invalid_argument("RoPE d_model must match attention d_model");
    }
}

Tensor MultiHeadAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value, 
                                  const Tensor* mask, bool training) {
    
    // Validate input dimensions [batch_size, seq_len, d_model]
    if (query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3) {
        throw std::invalid_argument("Input tensors must have 3 dimensions [batch_size, seq_len, d_model]");
    }
    if (query.shape()[2] != d_model_ || key.shape()[2] != d_model_ || value.shape()[2] != d_model_) {
        throw std::invalid_argument("Input tensor last dimension must match d_model");
    }
    
    size_t batch_size = query.shape()[0];
    size_t seq_len_q = query.shape()[1];
    size_t seq_len_k = key.shape()[1];
    size_t seq_len_v = value.shape()[1];
    
    // Linear transformations: Q, K, V
    Tensor Q = linear_transform(query, *W_q_, *b_q_);
    Tensor K = linear_transform(key, *W_k_, *b_k_);
    Tensor V = linear_transform(value, *W_v_, *b_v_);
    
    // Apply RoPE if enabled
    if (use_rope_ && rope_encoding_) {
        auto [Q_rope, K_rope] = rope_encoding_->apply_rotary_pos_emb_qk(Q, K, seq_len_q);
        Q = Q_rope;
        K = K_rope;
        // V doesn't get rotated in RoPE
    }
    
    // Reshape for multi-head attention: [batch_size, n_heads, seq_len, d_k]
    Q = split_heads(Q, batch_size, seq_len_q);
    K = split_heads(K, batch_size, seq_len_k);
    V = split_heads(V, batch_size, seq_len_v);
    
    // Scaled dot-product attention
    Tensor attention_output = scaled_dot_product_attention(Q, K, V, mask, training);
    
    // Combine heads: [batch_size, seq_len, d_model]
    attention_output = combine_heads(attention_output, batch_size, seq_len_q);
    
    // Final linear transformation
    return linear_transform(attention_output, *W_o_, *b_o_);
}

Tensor MultiHeadAttention::scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                                       const Tensor* mask, bool training) {
    
    // Q, K, V shapes: [batch_size, n_heads, seq_len, d_k]
    size_t batch_size = Q.shape()[0];
    size_t n_heads = Q.shape()[1];
    size_t seq_len_q = Q.shape()[2];
    size_t seq_len_k = K.shape()[2];
    size_t d_k = Q.shape()[3];
    
    // Compute attention scores: QK^T / sqrt(d_k)
    // K transpose: [batch_size, n_heads, d_k, seq_len_k]
    Tensor K_T = K.transpose(-2, -1);
    
    // Matrix multiplication: [batch_size, n_heads, seq_len_q, seq_len_k]
    Tensor scores({batch_size, n_heads, seq_len_q, seq_len_k});
    
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    
    // Compute QK^T scaled
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < n_heads; ++h) {
            for (size_t i = 0; i < seq_len_q; ++i) {
                for (size_t j = 0; j < seq_len_k; ++j) {
                    float score = 0.0f;
                    for (size_t k = 0; k < d_k; ++k) {
                        score += Q[{b, h, i, k}] * K_T[{b, h, k, j}];
                    }
                    scores[{b, h, i, j}] = score * scale;
                }
            }
        }
    }
    
    // Apply mask if provided
    if (mask) {
        scores = apply_mask(scores, *mask);
    }
    
    // Apply softmax along last dimension
    Tensor attention_weights = scores.softmax(-1);
    
    // Apply dropout during training
    if (training && dropout_rate_ > 0.0f) {
        attention_weights = dropout(attention_weights, training);
    }
    
    // Apply attention to values: [batch_size, n_heads, seq_len_q, d_v]
    Tensor output({batch_size, n_heads, seq_len_q, d_k});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < n_heads; ++h) {
            for (size_t i = 0; i < seq_len_q; ++i) {
                for (size_t k = 0; k < d_k; ++k) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < seq_len_k; ++j) {
                        sum += attention_weights[{b, h, i, j}] * V[{b, h, j, k}];
                    }
                    output[{b, h, i, k}] = sum;
                }
            }
        }
    }
    
    return output;
}

Tensor MultiHeadAttention::linear_transform(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    // input: [batch_size, seq_len, d_model]
    // weight: [d_model, d_model]
    // bias: [d_model]
    // output: [batch_size, seq_len, d_model]
    
    size_t batch_size = input.shape()[0];
    size_t seq_len = input.shape()[1];
    size_t d_model = input.shape()[2];
    
    Tensor output({batch_size, seq_len, d_model});
    
    // Matrix multiplication with bias
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t out_dim = 0; out_dim < d_model; ++out_dim) {
                float sum = bias[{out_dim}];
                for (size_t in_dim = 0; in_dim < d_model; ++in_dim) {
                    sum += input[{b, s, in_dim}] * weight[{in_dim, out_dim}];
                }
                output[{b, s, out_dim}] = sum;
            }
        }
    }
    
    return output;
}

Tensor MultiHeadAttention::split_heads(const Tensor& x, size_t batch_size, size_t seq_len) {
    // Input: [batch_size, seq_len, d_model]
    // Output: [batch_size, n_heads, seq_len, d_k]
    
    Tensor output({batch_size, n_heads_, seq_len, d_k_});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < n_heads_; ++h) {
                for (size_t d = 0; d < d_k_; ++d) {
                    size_t input_dim = h * d_k_ + d;
                    output[{b, h, s, d}] = x[{b, s, input_dim}];
                }
            }
        }
    }
    
    return output;
}

Tensor MultiHeadAttention::combine_heads(const Tensor& x, size_t batch_size, size_t seq_len) {
    // Input: [batch_size, n_heads, seq_len, d_k]
    // Output: [batch_size, seq_len, d_model]
    
    Tensor output({batch_size, seq_len, d_model_});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < n_heads_; ++h) {
                for (size_t d = 0; d < d_k_; ++d) {
                    size_t output_dim = h * d_k_ + d;
                    output[{b, s, output_dim}] = x[{b, h, s, d}];
                }
            }
        }
    }
    
    return output;
}

Tensor MultiHeadAttention::apply_mask(const Tensor& scores, const Tensor& mask) {
    // Apply mask by setting masked positions to large negative value
    Tensor masked_scores = scores;
    
    const float NEG_INF = -1e9f;
    
    // Assume mask shape matches scores or can be broadcasted
    size_t batch_size = scores.shape()[0];
    size_t n_heads = scores.shape()[1];
    size_t seq_len_q = scores.shape()[2];
    size_t seq_len_k = scores.shape()[3];
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < n_heads; ++h) {
            for (size_t i = 0; i < seq_len_q; ++i) {
                for (size_t j = 0; j < seq_len_k; ++j) {
                    // Check mask value (assuming binary mask)
                    float mask_val = 0.0f;
                    if (mask.ndim() == 2) {
                        mask_val = mask[{i, j}];
                    } else if (mask.ndim() == 4) {
                        mask_val = mask[{b, h, i, j}];
                    }
                    
                    if (mask_val < 0.5f) { // Masked position
                        masked_scores[{b, h, i, j}] = NEG_INF;
                    }
                }
            }
        }
    }
    
    return masked_scores;
}

Tensor MultiHeadAttention::dropout(const Tensor& x, bool training) {
    if (!training || dropout_rate_ <= 0.0f) {
        return x;
    }
    
    // Simple dropout implementation
    Tensor result = x;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    float keep_prob = 1.0f - dropout_rate_;
    float scale = 1.0f / keep_prob;
    
    // Apply dropout to all elements
    for (size_t i = 0; i < result.numel(); ++i) {
        if (dis(gen) < keep_prob) {
            result.matrix().raw_data()[i] *= scale;
        } else {
            result.matrix().raw_data()[i] = 0.0f;
        }
    }
    
    return result;
}

void MultiHeadAttention::init_parameters() {
    // Xavier/Glorot initialization for weights
    float fan_in = static_cast<float>(d_model_);
    float fan_out = static_cast<float>(d_model_);
    
    W_q_->xavier_uniform_(static_cast<size_t>(fan_in), static_cast<size_t>(fan_out));
    W_k_->xavier_uniform_(static_cast<size_t>(fan_in), static_cast<size_t>(fan_out));
    W_v_->xavier_uniform_(static_cast<size_t>(fan_in), static_cast<size_t>(fan_out));
    W_o_->xavier_uniform_(static_cast<size_t>(fan_in), static_cast<size_t>(fan_out));
    
    // Initialize biases to zero
    b_q_->zero_();
    b_k_->zero_();
    b_v_->zero_();
    b_o_->zero_();
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    return {W_q_.get(), W_k_.get(), W_v_.get(), W_o_.get(),
            b_q_.get(), b_k_.get(), b_v_.get(), b_o_.get()};
}