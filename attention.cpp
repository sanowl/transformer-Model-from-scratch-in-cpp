#include "attention.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t n_heads, float dropout)
    : d_model_(d_model), n_heads_(n_heads), dropout_rate_(dropout) {
    
    if (d_model % n_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by n_heads");
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
    Tensor output = input.matmul(weight);
    
    for (size_t i = 0; i < output.matrix().rows(); ++i) {
        for (size_t j = 0; j < output.matrix().cols(); ++j) {
            output.matrix()(i, j) += bias.matrix()(0, j);
        }
    }
    
    return output;
}

Tensor MultiHeadAttention::split_heads(const Tensor& x, size_t batch_size, size_t seq_len) {
    std::vector<size_t> new_shape = {batch_size, seq_len, n_heads_, d_k_};
    Tensor reshaped = x.reshape(new_shape);
    
    std::vector<size_t> perm_shape = {batch_size, n_heads_, seq_len, d_k_};
    return reshaped.view(perm_shape);
}

Tensor MultiHeadAttention::combine_heads(const Tensor& x, size_t batch_size, size_t seq_len) {
    std::vector<size_t> new_shape = {batch_size, seq_len, d_model_};
    return x.view(new_shape);
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
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
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
    encoding_ = std::make_unique<Tensor>(std::vector<size_t>{max_seq_len, d_model});
    compute_encoding();
}

void PositionalEncoding::compute_encoding() {
    Matrix& pe = encoding_->matrix();
    
    for (size_t pos = 0; pos < max_seq_len_; ++pos) {
        for (size_t i = 0; i < d_model_; ++i) {
            float angle = static_cast<float>(pos) / std::pow(10000.0f, (2.0f * (i / 2)) / static_cast<float>(d_model_));
            
            if (i % 2 == 0) {
                pe(pos, i) = std::sin(angle);
            } else {
                pe(pos, i) = std::cos(angle);
            }
        }
    }
}

Tensor PositionalEncoding::forward(const Tensor& x) {
    size_t seq_len = x.shape()[0];
    
    if (seq_len > max_seq_len_) {
        throw std::invalid_argument("Sequence length exceeds maximum positional encoding length");
    }
    
    Tensor pe_slice = encoding_->view({seq_len, d_model_});
    return x + pe_slice;
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
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
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