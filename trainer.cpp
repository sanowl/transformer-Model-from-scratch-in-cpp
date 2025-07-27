#include "trainer.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>

Trainer::Trainer(const TrainingConfig& config) 
    : config_(config), current_step_(0), best_loss_(std::numeric_limits<float>::max()) {
    
    tokenizer_ = std::make_unique<SimpleTokenizer>();
    model_ = std::make_unique<TransformerModel>(
        config_.vocab_size, config_.d_model, config_.n_heads,
        config_.n_layers, config_.d_ff, config_.max_seq_len, config_.dropout
    );
    optimizer_ = std::make_unique<AdamOptimizer>(config_.learning_rate);
}

void Trainer::prepare_data(const std::vector<std::string>& texts) {
    std::cout << "Building vocabulary..." << std::endl;
    tokenizer_->build_vocab(texts);
    
    size_t actual_vocab_size = tokenizer_->vocab_size();
    if (actual_vocab_size != config_.vocab_size) {
        std::cout << "Adjusting vocab size from " << config_.vocab_size 
                  << " to " << actual_vocab_size << std::endl;
        config_.vocab_size = actual_vocab_size;
        
        model_ = std::make_unique<TransformerModel>(
            config_.vocab_size, config_.d_model, config_.n_heads,
            config_.n_layers, config_.d_ff, config_.max_seq_len, config_.dropout
        );
    }
    
    std::cout << "Vocabulary size: " << tokenizer_->vocab_size() << std::endl;
}

std::vector<std::vector<size_t>> Trainer::prepare_sequences(const std::vector<std::string>& texts) {
    std::vector<std::vector<size_t>> sequences;
    
    for (const auto& text : texts) {
        auto tokens = tokenizer_->encode(text);
        if (tokens.size() <= config_.max_seq_len && tokens.size() > 1) {
            sequences.push_back(tokens);
        }
    }
    
    return sequences;
}

std::vector<std::vector<std::vector<size_t>>> Trainer::create_batches(const std::vector<std::vector<size_t>>& sequences) {
    std::vector<std::vector<std::vector<size_t>>> batches;
    
    for (size_t i = 0; i < sequences.size(); i += config_.batch_size) {
        std::vector<std::vector<size_t>> batch;
        for (size_t j = i; j < std::min(i + config_.batch_size, sequences.size()); ++j) {
            batch.push_back(sequences[j]);
        }
        batches.push_back(batch);
    }
    
    return batches;
}

void Trainer::backward_pass(const Tensor& logits, const std::vector<size_t>& targets) {
    auto params = model_->parameters();
    
    for (auto* param : params) {
        if (param->requires_grad()) {
            param->init_grad();
        }
    }
    
    Tensor output_grad = CrossEntropyLoss::compute_gradients(logits, targets);
    
    for (size_t i = 0; i < logits.matrix().rows(); ++i) {
        for (size_t j = 0; j < logits.matrix().cols(); ++j) {
            if (params.back()->has_grad()) {
                params.back()->grad().matrix()(j, i) += output_grad.matrix()(i, j);
            }
        }
    }
}

void Trainer::update_parameters() {
    auto params = model_->parameters();
    optimizer_->step(params);
    optimizer_->zero_grad(params);
}

void Trainer::train_epoch(const std::vector<std::vector<size_t>>& sequences) {
    auto batches = create_batches(sequences);
    float total_loss = 0.0f;
    size_t num_samples = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
        float batch_loss = 0.0f;
        
        for (const auto& sequence : batches[batch_idx]) {
            if (sequence.size() <= 1) continue;
            
            Tensor logits = model_->forward(sequence, nullptr, true);
            float loss = CrossEntropyLoss::compute_loss(logits, sequence);
            
            backward_pass(logits, sequence);
            
            batch_loss += loss;
            num_samples++;
        }
        
        update_parameters();
        total_loss += batch_loss;
        
        if (batch_idx % 10 == 0) {
            float avg_loss = batch_loss / batches[batch_idx].size();
            std::cout << "Batch " << batch_idx << "/" << batches.size() 
                      << ", Loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
        }
        
        current_step_++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    float avg_loss = total_loss / num_samples;
    std::cout << "Epoch completed. Average loss: " << std::fixed << std::setprecision(4) 
              << avg_loss << ", Time: " << duration.count() << "ms" << std::endl;
}

void Trainer::train(const std::vector<std::string>& texts) {
    prepare_data(texts);
    auto sequences = prepare_sequences(texts);
    
    std::cout << "Training on " << sequences.size() << " sequences" << std::endl;
    std::cout << "Model parameters: " << model_->parameters().size() << std::endl;
    
    for (size_t epoch = 0; epoch < config_.num_epochs; ++epoch) {
        std::cout << "\n=== Epoch " << (epoch + 1) << "/" << config_.num_epochs << " ===" << std::endl;
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(sequences.begin(), sequences.end(), g);
        
        train_epoch(sequences);
        
        if (epoch % 2 == 0) {
            float eval_loss = evaluate(sequences);
            std::cout << "Evaluation loss: " << std::fixed << std::setprecision(4) << eval_loss << std::endl;
            
            if (eval_loss < best_loss_) {
                best_loss_ = eval_loss;
                std::cout << "New best model! Loss: " << best_loss_ << std::endl;
            }
        }
    }
}

float Trainer::evaluate(const std::vector<std::vector<size_t>>& sequences) {
    float total_loss = 0.0f;
    size_t num_samples = 0;
    
    for (const auto& sequence : sequences) {
        if (sequence.size() <= 1) continue;
        
        Tensor logits = model_->forward(sequence, nullptr, false);
        float loss = CrossEntropyLoss::compute_loss(logits, sequence);
        
        total_loss += loss;
        num_samples++;
        
        if (num_samples >= 100) break;
    }
    
    return total_loss / num_samples;
}

std::string Trainer::generate_text(const std::string& prompt, size_t max_length) {
    auto tokens = tokenizer_->encode(prompt);
    
    for (size_t i = 0; i < max_length; ++i) {
        if (tokens.size() >= config_.max_seq_len) break;
        
        Tensor logits = model_->forward(tokens, nullptr, false);
        
        size_t last_pos = tokens.size() - 1;
        std::vector<float> probs(logits.matrix().cols());
        
        float max_logit = logits.matrix()(last_pos, 0);
        for (size_t j = 1; j < logits.matrix().cols(); ++j) {
            max_logit = std::max(max_logit, logits.matrix()(last_pos, j));
        }
        
        float sum_exp = 0.0f;
        for (size_t j = 0; j < logits.matrix().cols(); ++j) {
            probs[j] = std::exp(logits.matrix()(last_pos, j) - max_logit);
            sum_exp += probs[j];
        }
        
        for (size_t j = 0; j < probs.size(); ++j) {
            probs[j] /= sum_exp;
        }
        
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::discrete_distribution<> dis(probs.begin(), probs.end());
        size_t next_token = dis(gen);
        
        if (next_token == tokenizer_->eos_token_id()) break;
        
        tokens.push_back(next_token);
    }
    
    return tokenizer_->decode(tokens);
}

void Trainer::save_model(const std::string& path) {
    std::cout << "Model saving not implemented yet" << std::endl;
}

void Trainer::load_model(const std::string& path) {
    std::cout << "Model loading not implemented yet" << std::endl;
}

std::vector<std::string> DataLoader::load_text_file(const std::string& filepath) {
    std::vector<std::string> texts;
    std::ifstream file(filepath);
    std::string line;
    
    while (std::getline(file, line)) {
        if (!line.empty()) {
            texts.push_back(line);
        }
    }
    
    return texts;
}

std::vector<std::string> DataLoader::create_sample_data() {
    return {
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a simple sentence.",
        "Machine learning is fascinating and powerful.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models can learn complex patterns from data.",
        "Transformers have revolutionized the field of artificial intelligence.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Self-supervised learning helps models learn from unlabeled data.",
        "Large language models demonstrate impressive capabilities.",
        "Neural networks are inspired by biological brain structures.",
        "Training deep models requires significant computational resources.",
        "Fine-tuning pre-trained models is an effective transfer learning approach.",
        "Gradient descent optimizes model parameters during training.",
        "Backpropagation computes gradients efficiently in neural networks.",
        "Regularization techniques help prevent overfitting in machine learning.",
        "Cross-validation is important for evaluating model performance.",
        "Feature engineering plays a crucial role in traditional machine learning.",
        "Ensemble methods combine multiple models for better predictions.",
        "Hyperparameter tuning is essential for optimal model performance.",
        "Data preprocessing significantly impacts model quality and effectiveness."
    };
}

void DataLoader::save_sequences(const std::vector<std::vector<size_t>>& sequences, const std::string& filepath) {
    std::ofstream file(filepath);
    for (const auto& seq : sequences) {
        for (size_t i = 0; i < seq.size(); ++i) {
            file << seq[i];
            if (i < seq.size() - 1) file << " ";
        }
        file << "\n";
    }
}

std::vector<std::vector<size_t>> DataLoader::load_sequences(const std::string& filepath) {
    std::vector<std::vector<size_t>> sequences;
    std::ifstream file(filepath);
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<size_t> seq;
        size_t token;
        
        while (iss >> token) {
            seq.push_back(token);
        }
        
        if (!seq.empty()) {
            sequences.push_back(seq);
        }
    }
    
    return sequences;
}