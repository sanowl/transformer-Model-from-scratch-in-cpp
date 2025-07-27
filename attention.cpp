#include "attention.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t n_heads, float dropout)
    : d_model_(d_model), n_heads_(n_heads), dropout_rate_(dropout) {
    
    // Input validation
    if (d_model == 0) {
        throw std::invalid_argument("d_model must be positive");
    }
    if (n_heads == 0) {
        throw std::invalid_argument("n_heads must be positive");
    }
    if (d_model % n_heads != 0) {
        throw std::invalid_argument("d_model (" + std::to_string(d_model) + 
                                  ") must be divisible by n_heads (" + std::to_string(n_heads) + ")");
    }
    if (dropout < 0.0f || dropout > 1.0f) {
        throw std::invalid_argument("dropout rate must be between 0 and 1, got: " + std::to_string(dropout));
    }
    
    d_k_ = d_model / n_heads;
    d_v_ = d_model / n_heads;
    
    W_q_ = std::make_unique<Tensor>(std::vector<size_t>{d_model, d_model}, true);
    W_k_ = std::make_unique<Tensor>(std::vector<size_t>{d_model, d_model}, true);
    W_v_ = std::make_unique<Tensor>(std::vector<size_t>{d_model, d_model}, true);
    W_o_ = std::make_unique<Tensor>(std::vector<size_t>{d_model, d_model}, true);
    
    b_q_ = std::make_unique<Tensor>(std::vector<size_t>{1, d_model}, true);
    b_k_ = std::make_unique<Tensor>(std::vector<size_t>{1, d_model}, true);
    b_v_ = std::make_unique<Tensor>(std::vector<size_t>{1, d_model}, true);
    b_o_ = std::make_unique<Tensor>(std::vector<size_t>{1, d_model}, true);
    
    init_parameters();
}

void MultiHeadAttention::init_parameters() {
    float scale = 1.0f / std::sqrt(static_cast<float>(d_model_));
    
    W_q_->xavier_uniform_(d_model_, d_model_);
    W_k_->xavier_uniform_(d_model_, d_model_);
    W_v_->xavier_uniform_(d_model_, d_model_);
    W_o_->xavier_uniform_(d_model_, d_model_);
    
    b_q_->zero_();
    b_k_->zero_();
    b_v_->zero_();
    b_o_->zero_();
}

Tensor MultiHeadAttention::linear_transform(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    // Validate input dimensions
    if (input.shape().size() != 2 || weight.shape().size() != 2 || bias.shape().size() != 2) {
        throw TensorShapeError("linear_transform requires 2D tensors");
    }
    
    if (input.shape()[1] != weight.shape()[0]) {
        throw TensorShapeError("Input dimension mismatch for linear transform. "
                             "Input cols: " + std::to_string(input.shape()[1]) + 
                             ", Weight rows: " + std::to_string(weight.shape()[0]));
    }
    
    if (weight.shape()[1] != bias.shape()[1]) {
        throw TensorShapeError("Weight and bias dimension mismatch");
    }
    
    try {
        Tensor output = input.matmul(weight);
        
        // Add bias with broadcasting
        for (size_t i = 0; i < output.matrix().rows(); ++i) {
            for (size_t j = 0; j < output.matrix().cols(); ++j) {
                output.matrix()(i, j) += bias.matrix()(0, j);
            }
        }
        
        return output;
    } catch (const std::exception& e) {
        throw TensorMemoryError("Failed in linear transformation: " + std::string(e.what()));
    }
}

Tensor MultiHeadAttention::split_heads(const Tensor& x, size_t batch_size, size_t seq_len) {
    // Input tensor should have shape [seq_len, d_model]
    // We need to reshape to [batch_size, seq_len, n_heads, d_k] then permute to [batch_size, n_heads, seq_len, d_k]
    
    // First, ensure we have the correct input shape
    if (x.shape().size() != 2 || x.shape()[0] != seq_len || x.shape()[1] != d_model_) {
        throw std::invalid_argument("Input tensor has incorrect shape for split_heads");
    }
    
    // Create a new tensor with the correct shape for multi-head attention
    // For simplicity, we'll work with [seq_len, n_heads, d_k] and treat batch_size as 1
    std::vector<size_t> new_shape = {seq_len, n_heads_, d_k_};
    Tensor result(new_shape);
    
    // Reshape the input from [seq_len, d_model] to [seq_len, n_heads, d_k]
    for (size_t seq = 0; seq < seq_len; ++seq) {
        for (size_t head = 0; head < n_heads_; ++head) {
            for (size_t k = 0; k < d_k_; ++k) {
                size_t input_col = head * d_k_ + k;
                result.matrix()(seq * n_heads_ + head, k) = x.matrix()(seq, input_col);
            }
        }
    }
    
    return result;
}

Tensor MultiHeadAttention::combine_heads(const Tensor& x, size_t batch_size, size_t seq_len) {
    // Input tensor should have shape [seq_len * n_heads, d_k]
    // We need to combine back to [seq_len, d_model]
    
    std::vector<size_t> result_shape = {seq_len, d_model_};
    Tensor result(result_shape);
    
    // Combine the heads back to original shape
    for (size_t seq = 0; seq < seq_len; ++seq) {
        for (size_t head = 0; head < n_heads_; ++head) {
            for (size_t k = 0; k < d_k_; ++k) {
                size_t output_col = head * d_k_ + k;
                result.matrix()(seq, output_col) = x.matrix()(seq * n_heads_ + head, k);
            }
        }
    }
    
    return result;
}

Tensor MultiHeadAttention::apply_mask(const Tensor& scores, const Tensor& mask) {
    Tensor masked_scores = scores;
    
    const float neg_inf = -1e9f;
    for (size_t i = 0; i < scores.matrix().rows(); ++i) {
        for (size_t j = 0; j < scores.matrix().cols(); ++j) {
            size_t mask_i = i % mask.matrix().rows();
            size_t mask_j = j % mask.matrix().cols();
            
            if (mask.matrix()(mask_i, mask_j) == 0.0f) {
                masked_scores.matrix()(i, j) = neg_inf;
            }
        }
    }
    
    return masked_scores;
}

Tensor MultiHeadAttention::dropout(const Tensor& x, bool training) {
    if (!training || dropout_rate_ == 0.0f) {
        return x;
    }
    
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::bernoulli_distribution dis(1.0f - dropout_rate_);
    
    Tensor result = x;
    float scale = 1.0f / (1.0f - dropout_rate_);
    
    for (size_t i = 0; i < result.matrix().rows(); ++i) {
        for (size_t j = 0; j < result.matrix().cols(); ++j) {
            if (dis(gen)) {
                result.matrix()(i, j) *= scale;
            } else {
                result.matrix()(i, j) = 0.0f;
            }
        }
    }
    
    return result;
}

Tensor MultiHeadAttention::scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                                       const Tensor* mask, bool training) {
    // Input validation
    if (Q.shape().size() != 2 || K.shape().size() != 2 || V.shape().size() != 2) {
        throw TensorShapeError("Attention inputs must be 2D tensors");
    }
    
    if (Q.shape()[1] != K.shape()[1]) {
        throw TensorShapeError("Q and K must have same hidden dimension");
    }
    
    if (K.shape()[0] != V.shape()[0]) {
        throw TensorShapeError("K and V must have same sequence length");
    }
    
    if (d_k_ == 0) {
        throw std::runtime_error("d_k_ is zero, cannot compute attention scale");
    }
    
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k_));
    
    Tensor K_transposed = K.transpose();
    Tensor scores = Q.matmul(K_transposed);
    scores *= scale;

    if (mask) {
        scores = apply_mask(scores, *mask);
    }
    
    Tensor attention_weights = scores.softmax(-1);
    attention_weights = dropout(attention_weights, training);
    
    Tensor output = attention_weights.matmul(V);
    
    return output;
}

Tensor MultiHeadAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value,
                                  const Tensor* mask, bool training) {
    
    size_t batch_size = 1;
    size_t seq_len = query.shape()[0];
    
    Tensor Q = linear_transform(query, *W_q_, *b_q_);
    Tensor K = linear_transform(key, *W_k_, *b_k_);
    Tensor V = linear_transform(value, *W_v_, *b_v_);
    
    Q = split_heads(Q, batch_size, seq_len);
    K = split_heads(K, batch_size, seq_len);
    V = split_heads(V, batch_size, seq_len);
    
    Tensor attention_output = scaled_dot_product_attention(Q, K, V, mask, training);
    
    attention_output = combine_heads(attention_output, batch_size, seq_len);
    
    Tensor output = linear_transform(attention_output, *W_o_, *b_o_);
    
    return output;
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    return {W_q_.get(), W_k_.get(), W_v_.get(), W_o_.get(),
            b_q_.get(), b_k_.get(), b_v_.get(), b_o_.get()};
}

PositionalEncoding::PositionalEncoding(size_t d_model, size_t max_seq_len)
    : d_model_(d_model), max_seq_len_(max_seq_len) {
    
    // Input validation
    if (d_model == 0) {
        throw std::invalid_argument("d_model must be positive");
    }
    if (max_seq_len == 0) {
        throw std::invalid_argument("max_seq_len must be positive");
    }
    if (max_seq_len > 1000000) { // Reasonable upper limit
        throw std::invalid_argument("max_seq_len too large: " + std::to_string(max_seq_len));
    }
    
    try {
        encoding_ = std::make_unique<Tensor>(std::vector<size_t>{max_seq_len, d_model});
        compute_encoding();
    } catch (const std::exception& e) {
        throw TensorMemoryError("Failed to create positional encoding: " + std::string(e.what()));
    }
}

void PositionalEncoding::compute_encoding() {
    Matrix& pe = encoding_->matrix();
    
    for (size_t pos = 0; pos < max_seq_len_; ++pos) {
        for (size_t i = 0; i < d_model_; ++i) {
            // Fix integer division issue: ensure floating point division
            float angle = static_cast<float>(pos) / std::pow(10000.0f, (2.0f * static_cast<float>(i / 2)) / static_cast<float>(d_model_));
            
            if (i % 2 == 0) {
                pe(pos, i) = std::sin(angle);
            } else {
                pe(pos, i) = std::cos(angle);
            }
        }
    }
}

Tensor PositionalEncoding::forward(const Tensor& x) {
    // Input validation
    if (x.shape().size() != 2) {
        throw TensorShapeError("PositionalEncoding input must be 2D tensor");
    }
    
    size_t seq_len = x.shape()[0];
    size_t input_d_model = x.shape()[1];
    
    if (seq_len == 0) {
        throw TensorShapeError("Sequence length cannot be zero");
    }
    
    if (input_d_model != d_model_) {
        throw TensorShapeError("Input d_model (" + std::to_string(input_d_model) + 
                             ") doesn't match expected (" + std::to_string(d_model_) + ")");
    }
    
    if (seq_len > max_seq_len_) {
        throw TensorShapeError("Sequence length (" + std::to_string(seq_len) + 
                             ") exceeds maximum (" + std::to_string(max_seq_len_) + ")");
    }
    
    try {
        Tensor pe_slice = encoding_->view({seq_len, d_model_});
        return x + pe_slice;
    } catch (const std::exception& e) {
        throw TensorMemoryError("Failed in positional encoding forward pass: " + std::string(e.what()));
    }
}

LayerNorm::LayerNorm(size_t d_model, float eps)
    : d_model_(d_model), eps_(eps) {
    gamma_ = std::make_unique<Tensor>(std::vector<size_t>{1, d_model}, true);
    beta_ = std::make_unique<Tensor>(std::vector<size_t>{1, d_model}, true);
    init_parameters();
}

void LayerNorm::init_parameters() {
    gamma_->fill_(1.0f);
    beta_->zero_();
}

Tensor LayerNorm::forward(const Tensor& x) {
    Tensor mean = x.mean(1, true);
    Tensor var = x.var(1, true);
    
    Tensor normalized = (x - mean);
    
    for (size_t i = 0; i < normalized.matrix().rows(); ++i) {
        float std_dev = std::sqrt(var.matrix()(i, 0) + eps_);
        for (size_t j = 0; j < normalized.matrix().cols(); ++j) {
            normalized.matrix()(i, j) /= std_dev;
        }
    }
    
    Tensor output = normalized;
    for (size_t i = 0; i < output.matrix().rows(); ++i) {
        for (size_t j = 0; j < output.matrix().cols(); ++j) {
            output.matrix()(i, j) = output.matrix()(i, j) * gamma_->matrix()(0, j) + beta_->matrix()(0, j);
        }
    }
    
    return output;
}

std::vector<Tensor*> LayerNorm::parameters() {
    return {gamma_.get(), beta_.get()};
}

FeedForward::FeedForward(size_t d_model, size_t d_ff, float dropout)
    : d_model_(d_model), d_ff_(d_ff), dropout_rate_(dropout) {
    
    W1_ = std::make_unique<Tensor>(std::vector<size_t>{d_model, d_ff}, true);
    W2_ = std::make_unique<Tensor>(std::vector<size_t>{d_ff, d_model}, true);
    b1_ = std::make_unique<Tensor>(std::vector<size_t>{1, d_ff}, true);
    b2_ = std::make_unique<Tensor>(std::vector<size_t>{1, d_model}, true);
    
    init_parameters();
}

void FeedForward::init_parameters() {
    W1_->xavier_uniform_(d_model_, d_ff_);
    W2_->xavier_uniform_(d_ff_, d_model_);
    b1_->zero_();
    b2_->zero_();
}

Tensor FeedForward::dropout(const Tensor& x, bool training) {
    if (!training || dropout_rate_ == 0.0f) {
        return x;
    }
    
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::bernoulli_distribution dis(1.0f - dropout_rate_);
    
    Tensor result = x;
    float scale = 1.0f / (1.0f - dropout_rate_);
    
    for (size_t i = 0; i < result.matrix().rows(); ++i) {
        for (size_t j = 0; j < result.matrix().cols(); ++j) {
            if (dis(gen)) {
                result.matrix()(i, j) *= scale;
            } else {
                result.matrix()(i, j) = 0.0f;
            }
        }
    }
    
    return result;
}

Tensor FeedForward::forward(const Tensor& x, bool training) {
    Tensor hidden = x.matmul(*W1_);
    
    for (size_t i = 0; i < hidden.matrix().rows(); ++i) {
        for (size_t j = 0; j < hidden.matrix().cols(); ++j) {
            hidden.matrix()(i, j) += b1_->matrix()(0, j);
        }
    }
    
    hidden = hidden.gelu();
    hidden = dropout(hidden, training);
    
    Tensor output = hidden.matmul(*W2_);
    
    for (size_t i = 0; i < output.matrix().rows(); ++i) {
        for (size_t j = 0; j < output.matrix().cols(); ++j) {
            output.matrix()(i, j) += b2_->matrix()(0, j);
        }
    }
    
    return dropout(output, training);
}

std::vector<Tensor*> FeedForward::parameters() {
    return {W1_.get(), W2_.get(), b1_.get(), b2_.get()};
}