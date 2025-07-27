#include "transformer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <cmath>

TransformerBlock::TransformerBlock(size_t d_model, size_t n_heads, size_t d_ff, float dropout)
    : d_model_(d_model), n_heads_(n_heads), d_ff_(d_ff), dropout_rate_(dropout) {
    
    self_attention_ = std::make_unique<MultiHeadAttention>(d_model, n_heads, dropout);
    norm1_ = std::make_unique<LayerNorm>(d_model);
    feed_forward_ = std::make_unique<FeedForward>(d_model, d_ff, dropout);
    norm2_ = std::make_unique<LayerNorm>(d_model);
}

Tensor TransformerBlock::forward(const Tensor& x, const Tensor* mask, bool training) {
    Tensor attn_output = self_attention_->forward(x, x, x, mask, training);
    Tensor x1 = norm1_->forward(x + attn_output);
    
    Tensor ff_output = feed_forward_->forward(x1, training);
    Tensor x2 = norm2_->forward(x1 + ff_output);
    
    return x2;
}

std::vector<Tensor*> TransformerBlock::parameters() {
    std::vector<Tensor*> params;
    
    auto attn_params = self_attention_->parameters();
    params.insert(params.end(), attn_params.begin(), attn_params.end());
    
    auto norm1_params = norm1_->parameters();
    params.insert(params.end(), norm1_params.begin(), norm1_params.end());
    
    auto ff_params = feed_forward_->parameters();
    params.insert(params.end(), ff_params.begin(), ff_params.end());
    
    auto norm2_params = norm2_->parameters();
    params.insert(params.end(), norm2_params.begin(), norm2_params.end());
    
    return params;
}

TransformerModel::TransformerModel(size_t vocab_size, size_t d_model, size_t n_heads,
                                  size_t n_layers, size_t d_ff, size_t max_seq_len, float dropout)
    : vocab_size_(vocab_size), d_model_(d_model), n_heads_(n_heads), n_layers_(n_layers),
      d_ff_(d_ff), max_seq_len_(max_seq_len), dropout_rate_(dropout) {
    
    token_embedding_ = std::make_unique<Tensor>(std::vector<size_t>{vocab_size, d_model}, true);
    pos_encoding_ = std::make_unique<PositionalEncoding>(d_model, max_seq_len);
    
    for (size_t i = 0; i < n_layers; ++i) {
        layers_.push_back(std::make_unique<TransformerBlock>(d_model, n_heads, d_ff, dropout));
    }
    
    final_norm_ = std::make_unique<LayerNorm>(d_model);
    output_projection_ = std::make_unique<Tensor>(std::vector<size_t>{d_model, vocab_size}, true);
    
    init_parameters();
}

void TransformerModel::init_parameters() {
    init_embeddings();
    output_projection_->xavier_uniform_(d_model_, vocab_size_);
}

void TransformerModel::init_embeddings() {
    token_embedding_->normal_(0.0f, 0.02f);
}

Tensor TransformerModel::embedding_lookup(const std::vector<size_t>& input_ids) {
    if (input_ids.empty()) {
        throw TensorShapeError("Input IDs cannot be empty");
    }
    
    size_t seq_len = input_ids.size();
    if (seq_len > max_seq_len_) {
        throw TensorShapeError("Input sequence length (" + std::to_string(seq_len) + 
                             ") exceeds maximum (" + std::to_string(max_seq_len_) + ")");
    }
    
    try {
        Tensor embeddings(std::vector<size_t>{seq_len, d_model_});
        
        for (size_t i = 0; i < seq_len; ++i) {
            size_t token_id = input_ids[i];
            if (token_id >= vocab_size_) {
                throw TensorShapeError("Token ID (" + std::to_string(token_id) + 
                                     ") out of vocabulary range (0-" + std::to_string(vocab_size_ - 1) + ")");
            }
            
            for (size_t j = 0; j < d_model_; ++j) {
                embeddings.matrix()(i, j) = token_embedding_->matrix()(token_id, j);
            }
        }
        
        return embeddings;
    } catch (const std::exception& e) {
        throw TensorMemoryError("Embedding lookup failed: " + std::string(e.what()));
    }
}

Tensor TransformerModel::generate_causal_mask(size_t seq_len) {
    Tensor mask(std::vector<size_t>{seq_len, seq_len});
    
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            mask.matrix()(i, j) = (j <= i) ? 1.0f : 0.0f;
        }
    }
    
    return mask;
}

Tensor TransformerModel::forward(const std::vector<size_t>& input_ids, const Tensor* mask, bool training) {
    Tensor x = embedding_lookup(input_ids);
    x = pos_encoding_->forward(x);
    
    Tensor causal_mask = generate_causal_mask(input_ids.size());
    const Tensor* effective_mask = mask ? mask : &causal_mask;
    
    for (auto& layer : layers_) {
        x = layer->forward(x, effective_mask, training);
    }
    
    x = final_norm_->forward(x);
    
    Tensor logits = x.matmul(*output_projection_);
    return logits;
}

std::vector<Tensor*> TransformerModel::parameters() {
    std::vector<Tensor*> params;
    
    params.push_back(token_embedding_.get());
    
    for (auto& layer : layers_) {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    
    auto final_norm_params = final_norm_->parameters();
    params.insert(params.end(), final_norm_params.begin(), final_norm_params.end());
    
    params.push_back(output_projection_.get());
    
    return params;
}

SimpleTokenizer::SimpleTokenizer() : vocab_size_(0) {
    vocab_["<unk>"] = 0;
    vocab_["<pad>"] = 1;
    vocab_["<eos>"] = 2;
    vocab_["<bos>"] = 3;
    
    id_to_token_[0] = "<unk>";
    id_to_token_[1] = "<pad>";
    id_to_token_[2] = "<eos>";
    id_to_token_[3] = "<bos>";
    
    unk_token_id_ = 0;
    pad_token_id_ = 1;
    eos_token_id_ = 2;
    bos_token_id_ = 3;
    vocab_size_ = 4;
}

std::vector<std::string> SimpleTokenizer::tokenize_text(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string word;
    
    while (ss >> word) {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        std::string clean_word;
        for (char c : word) {
            if (std::isalnum(c)) {
                clean_word += c;
            }
        }
        
        if (!clean_word.empty()) {
            tokens.push_back(clean_word);
        }
    }
    
    return tokens;
}

void SimpleTokenizer::build_vocab(const std::vector<std::string>& texts) {
    std::unordered_map<std::string, size_t> word_counts;
    
    for (const auto& text : texts) {
        auto tokens = tokenize_text(text);
        for (const auto& token : tokens) {
            word_counts[token]++;
        }
    }
    
    std::vector<std::pair<std::string, size_t>> sorted_words(word_counts.begin(), word_counts.end());
    std::sort(sorted_words.begin(), sorted_words.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (const auto& word_count : sorted_words) {
        if (vocab_.find(word_count.first) == vocab_.end()) {
            vocab_[word_count.first] = vocab_size_;
            id_to_token_[vocab_size_] = word_count.first;
            vocab_size_++;
        }
    }
}

std::vector<size_t> SimpleTokenizer::encode(const std::string& text) {
    auto tokens = tokenize_text(text);
    std::vector<size_t> token_ids;
    
    token_ids.push_back(bos_token_id_);
    
    for (const auto& token : tokens) {
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(unk_token_id_);
        }
    }
    
    token_ids.push_back(eos_token_id_);
    return token_ids;
}

std::string SimpleTokenizer::decode(const std::vector<size_t>& token_ids) {
    std::string result;
    
    for (size_t token_id : token_ids) {
        if (token_id == pad_token_id_) continue;
        if (token_id == bos_token_id_) continue;
        if (token_id == eos_token_id_) break;
        
        auto it = id_to_token_.find(token_id);
        if (it != id_to_token_.end()) {
            if (!result.empty()) result += " ";
            result += it->second;
        }
    }
    
    return result;
}

float CrossEntropyLoss::compute_loss(const Tensor& logits, const std::vector<size_t>& targets) {
    float total_loss = 0.0f;
    size_t seq_len = targets.size();
    
    for (size_t t = 0; t < seq_len; ++t) {
        if (t + 1 < seq_len) {
            size_t target = targets[t + 1];
            
            float max_logit = logits.matrix()(t, 0);
            for (size_t j = 1; j < logits.matrix().cols(); ++j) {
                max_logit = std::max(max_logit, logits.matrix()(t, j));
            }
            
            float sum_exp = 0.0f;
            for (size_t j = 0; j < logits.matrix().cols(); ++j) {
                sum_exp += std::exp(logits.matrix()(t, j) - max_logit);
            }
            
            float log_prob = logits.matrix()(t, target) - max_logit - std::log(sum_exp);
            total_loss -= log_prob;
        }
    }
    
    return total_loss / static_cast<float>(seq_len - 1);
}

Tensor CrossEntropyLoss::compute_gradients(const Tensor& logits, const std::vector<size_t>& targets) {
    Tensor grad(std::vector<size_t>{logits.shape()[0], logits.shape()[1]});
    size_t seq_len = targets.size();
    
    for (size_t t = 0; t < seq_len - 1; ++t) {
        size_t target = targets[t + 1];
        
        Tensor probs = logits.view({1, logits.shape()[1]});
        Matrix& prob_matrix = probs.matrix();
        
        float max_logit = logits.matrix()(t, 0);
        for (size_t j = 1; j < logits.matrix().cols(); ++j) {
            max_logit = std::max(max_logit, logits.matrix()(t, j));
        }
        
        float sum_exp = 0.0f;
        for (size_t j = 0; j < logits.matrix().cols(); ++j) {
            prob_matrix(0, j) = std::exp(logits.matrix()(t, j) - max_logit);
            sum_exp += prob_matrix(0, j);
        }
        
        for (size_t j = 0; j < logits.matrix().cols(); ++j) {
            prob_matrix(0, j) /= sum_exp;
            grad.matrix()(t, j) = prob_matrix(0, j);
        }
        
        grad.matrix()(t, target) -= 1.0f;
        
        for (size_t j = 0; j < logits.matrix().cols(); ++j) {
            grad.matrix()(t, j) /= static_cast<float>(seq_len - 1);
        }
    }
    
    return grad;
}

AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), step_(0) {
}

void AdamOptimizer::initialize_moments(std::vector<Tensor*>& parameters) {
    if (m_.empty()) {
        for (auto* param : parameters) {
            m_.push_back(std::make_unique<Tensor>(param->shape()));
            v_.push_back(std::make_unique<Tensor>(param->shape()));
            m_.back()->zero_();
            v_.back()->zero_();
        }
    }
}

void AdamOptimizer::step(std::vector<Tensor*>& parameters) {
    initialize_moments(parameters);
    step_++;
    
    float lr_t = learning_rate_ * std::sqrt(1.0f - std::pow(beta2_, step_)) / (1.0f - std::pow(beta1_, step_));
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        if (!parameters[i]->has_grad()) continue;
        
        Tensor& grad = parameters[i]->grad();
        Tensor& m = *m_[i];
        Tensor& v = *v_[i];
        
        for (size_t j = 0; j < grad.matrix().rows(); ++j) {
            for (size_t k = 0; k < grad.matrix().cols(); ++k) {
                float g = grad.matrix()(j, k);
                
                m.matrix()(j, k) = beta1_ * m.matrix()(j, k) + (1.0f - beta1_) * g;
                v.matrix()(j, k) = beta2_ * v.matrix()(j, k) + (1.0f - beta2_) * g * g;
                
                float m_hat = m.matrix()(j, k);
                float v_hat = v.matrix()(j, k);
                
                parameters[i]->matrix()(j, k) -= lr_t * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }
}

void AdamOptimizer::zero_grad(std::vector<Tensor*>& parameters) {
    for (auto* param : parameters) {
        if (param->has_grad()) {
            param->zero_grad();
        }
    }
}