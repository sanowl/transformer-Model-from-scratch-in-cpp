#pragma once

#include "transformer.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <chrono>
#include <mutex>
#include <atomic>

struct TrainingConfig {
    size_t vocab_size = 10000;
    size_t d_model = 512;
    size_t n_heads = 8;
    size_t n_layers = 6;
    size_t d_ff = 2048;
    size_t max_seq_len = 512;
    float dropout = 0.1f;
    float learning_rate = 0.0001f;
    size_t batch_size = 32;
    size_t num_epochs = 10;
    size_t save_every = 1000;
    bool use_cuda = false;
};

class Trainer {
private:
    std::unique_ptr<TransformerModel> model_;
    std::unique_ptr<AdamOptimizer> optimizer_;
    std::unique_ptr<SimpleTokenizer> tokenizer_;
    TrainingConfig config_;
    
    size_t current_step_;
    float best_loss_;

public:
    Trainer(const TrainingConfig& config);
    
    void prepare_data(const std::vector<std::string>& texts);
    void train_epoch(const std::vector<std::vector<size_t>>& sequences);
    void train(const std::vector<std::string>& texts);
    
    float evaluate(const std::vector<std::vector<size_t>>& sequences);
    std::string generate_text(const std::string& prompt, size_t max_length = 100);
    
    void save_model(const std::string& path);
    void load_model(const std::string& path);

private:
    std::vector<std::vector<size_t>> prepare_sequences(const std::vector<std::string>& texts);
    std::vector<std::vector<std::vector<size_t>>> create_batches(const std::vector<std::vector<size_t>>& sequences);
    void backward_pass(const Tensor& logits, const std::vector<size_t>& targets);
    void update_parameters();
};

class DataLoader {
public:
    static std::vector<std::string> load_text_file(const std::string& filepath);
    static std::vector<std::string> create_sample_data();
    static void save_sequences(const std::vector<std::vector<size_t>>& sequences, const std::string& filepath);
    static std::vector<std::vector<size_t>> load_sequences(const std::string& filepath);
};