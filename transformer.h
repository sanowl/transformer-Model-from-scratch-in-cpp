#pragma once

#include "attention.h"
#include "tensor.h"
#include <vector>
#include <memory>

class TransformerBlock {
private:
    size_t d_model_;
    size_t n_heads_;
    size_t d_ff_;
    float dropout_rate_;
    
    std::unique_ptr<MultiHeadAttention> self_attention_;
    std::unique_ptr<LayerNorm> norm1_;
    std::unique_ptr<FeedForward> feed_forward_;
    std::unique_ptr<LayerNorm> norm2_;

public:
    TransformerBlock(size_t d_model, size_t n_heads, size_t d_ff, float dropout = 0.1f);
    
    Tensor forward(const Tensor& x, const Tensor* mask = nullptr, bool training = true);
    std::vector<Tensor*> parameters();
};

class TransformerModel {
private:
    size_t vocab_size_;
    size_t d_model_;
    size_t n_heads_;
    size_t n_layers_;
    size_t d_ff_;
    size_t max_seq_len_;
    float dropout_rate_;
    
    std::unique_ptr<Tensor> token_embedding_;
    std::unique_ptr<PositionalEncoding> pos_encoding_;
    std::vector<std::unique_ptr<TransformerBlock>> layers_;
    std::unique_ptr<LayerNorm> final_norm_;
    std::unique_ptr<Tensor> output_projection_;

public:
    TransformerModel(size_t vocab_size, size_t d_model, size_t n_heads, 
                    size_t n_layers, size_t d_ff, size_t max_seq_len = 512, 
                    float dropout = 0.1f);
    
    Tensor forward(const std::vector<size_t>& input_ids, const Tensor* mask = nullptr, bool training = true);
    Tensor generate_causal_mask(size_t seq_len);
    
    void init_parameters();
    std::vector<Tensor*> parameters();
    
    Tensor embedding_lookup(const std::vector<size_t>& input_ids);
    
private:
    void init_embeddings();
};

class SimpleTokenizer {
private:
    std::unordered_map<std::string, size_t> vocab_;
    std::unordered_map<size_t, std::string> id_to_token_;
    size_t vocab_size_;
    size_t unk_token_id_;
    size_t pad_token_id_;
    size_t eos_token_id_;
    size_t bos_token_id_;

public:
    SimpleTokenizer();
    
    void build_vocab(const std::vector<std::string>& texts);
    std::vector<size_t> encode(const std::string& text);
    std::string decode(const std::vector<size_t>& token_ids);
    
    size_t vocab_size() const { return vocab_size_; }
    size_t unk_token_id() const { return unk_token_id_; }
    size_t pad_token_id() const { return pad_token_id_; }
    size_t eos_token_id() const { return eos_token_id_; }
    size_t bos_token_id() const { return bos_token_id_; }
    
private:
    std::vector<std::string> tokenize_text(const std::string& text);
};

class CrossEntropyLoss {
public:
    static float compute_loss(const Tensor& logits, const std::vector<size_t>& targets);
    static Tensor compute_gradients(const Tensor& logits, const std::vector<size_t>& targets);
};

class AdamOptimizer {
private:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    size_t step_;
    
    std::vector<std::unique_ptr<Tensor>> m_;
    std::vector<std::unique_ptr<Tensor>> v_;

public:
    AdamOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f, 
                  float beta2 = 0.999f, float epsilon = 1e-8f);
    
    void step(std::vector<Tensor*>& parameters);
    void zero_grad(std::vector<Tensor*>& parameters);
    
private:
    void initialize_moments(std::vector<Tensor*>& parameters);
};