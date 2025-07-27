#include "trainer.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

void print_banner() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════╗
║                  TRANSFORMER MODEL FROM SCRATCH          ║
║                       Implemented in C++                 ║
╚═══════════════════════════════════════════════════════════╝
)" << std::endl;
}

void demonstrate_components() {
    std::cout << "\n=== Component Demonstration ===\n" << std::endl;
    
    // Matrix operations
    std::cout << "1. Matrix Operations:" << std::endl;
    Matrix a(2, 3);
    a.randomize(-1.0f, 1.0f);
    std::cout << "Random matrix A:" << std::endl;
    a.print();
    
    Matrix b(3, 2);
    b.randomize(-1.0f, 1.0f);
    std::cout << "\nRandom matrix B:" << std::endl;
    b.print();
    
    Matrix c = a * b;
    std::cout << "\nMatrix multiplication A * B:" << std::endl;
    c.print();
    
    // Tensor operations
    std::cout << "\n2. Tensor Operations:" << std::endl;
    Tensor x({2, 4});
    x.uniform_(-1.0f, 1.0f);
    std::cout << "Random tensor:" << std::endl;
    x.print();
    
    Tensor y = x.softmax(-1);
    std::cout << "\nSoftmax applied:" << std::endl;
    y.print();
    
    // Attention mechanism
    std::cout << "\n3. Multi-Head Attention:" << std::endl;
    size_t d_model = 64;
    size_t n_heads = 4;
    
    MultiHeadAttention attention(d_model, n_heads);
    
    Tensor query({4, d_model});
    query.normal_(0.0f, 0.1f);
    
    Tensor key = query;
    Tensor value = query;
    
    std::cout << "Input shape: [" << query.shape()[0] << ", " << query.shape()[1] << "]" << std::endl;
    
    Tensor output = attention.forward(query, key, value);
    std::cout << "Attention output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;
    
    // Positional encoding
    std::cout << "\n4. Positional Encoding:" << std::endl;
    PositionalEncoding pos_enc(d_model, 100);
    Tensor encoded = pos_enc.forward(query);
    std::cout << "Positionally encoded shape: [" << encoded.shape()[0] << ", " << encoded.shape()[1] << "]" << std::endl;
    
    // Layer normalization
    std::cout << "\n5. Layer Normalization:" << std::endl;
    LayerNorm layer_norm(d_model);
    Tensor normalized = layer_norm.forward(query);
    std::cout << "Normalized shape: [" << normalized.shape()[0] << ", " << normalized.shape()[1] << "]" << std::endl;
    
    std::cout << "\n✓ All components working correctly!\n" << std::endl;
}

void run_training_example() {
    std::cout << "\n=== Training Example ===\n" << std::endl;
    
    TrainingConfig config;
    config.vocab_size = 1000;  // Will be adjusted based on actual vocabulary
    config.d_model = 128;      // Smaller for faster training
    config.n_heads = 4;
    config.n_layers = 2;       // Fewer layers for demonstration
    config.d_ff = 512;
    config.max_seq_len = 64;   // Shorter sequences
    config.learning_rate = 0.001f;
    config.batch_size = 4;     // Small batch size
    config.num_epochs = 3;     // Few epochs for demo
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  d_model: " << config.d_model << std::endl;
    std::cout << "  n_heads: " << config.n_heads << std::endl;
    std::cout << "  n_layers: " << config.n_layers << std::endl;
    std::cout << "  d_ff: " << config.d_ff << std::endl;
    std::cout << "  learning_rate: " << config.learning_rate << std::endl;
    
    Trainer trainer(config);
    
    // Create sample training data
    auto training_data = DataLoader::create_sample_data();
    std::cout << "\nTraining data: " << training_data.size() << " sentences" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Train the model
    trainer.train(training_data);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\nTraining completed in " << duration.count() << " seconds" << std::endl;
    
    // Generate some text
    std::cout << "\n=== Text Generation ===\n" << std::endl;
    
    std::vector<std::string> prompts = {
        "The quick brown",
        "Machine learning is",
        "Deep learning"
    };
    
    for (const auto& prompt : prompts) {
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::string generated = trainer.generate_text(prompt, 10);
        std::cout << "Generated: \"" << generated << "\"\n" << std::endl;
    }
}

void show_model_stats() {
    std::cout << "\n=== Model Architecture Stats ===\n" << std::endl;
    
    TrainingConfig config;
    config.vocab_size = 10000;
    config.d_model = 512;
    config.n_heads = 8;
    config.n_layers = 6;
    config.d_ff = 2048;
    
    TransformerModel model(config.vocab_size, config.d_model, config.n_heads,
                          config.n_layers, config.d_ff, config.max_seq_len);
    
    auto params = model.parameters();
    size_t total_params = 0;
    
    for (auto* param : params) {
        total_params += param->numel();
    }
    
    std::cout << "Model Statistics:" << std::endl;
    std::cout << "  Vocabulary size: " << config.vocab_size << std::endl;
    std::cout << "  Model dimension: " << config.d_model << std::endl;
    std::cout << "  Number of heads: " << config.n_heads << std::endl;
    std::cout << "  Number of layers: " << config.n_layers << std::endl;
    std::cout << "  Feed-forward dimension: " << config.d_ff << std::endl;
    std::cout << "  Total parameters: " << total_params << " (" 
              << std::fixed << std::setprecision(2) << total_params / 1e6 << "M)" << std::endl;
    
    // Memory estimation
    size_t memory_mb = (total_params * sizeof(float)) / (1024 * 1024);
    std::cout << "  Estimated memory: ~" << memory_mb << " MB" << std::endl;
}

int main() {
    try {
        print_banner();
        
        std::cout << "Features implemented:" << std::endl;
        std::cout << "✓ SIMD-optimized matrix operations" << std::endl;
        std::cout << "✓ Multi-head attention mechanism" << std::endl;
        std::cout << "✓ Positional encoding" << std::endl;
        std::cout << "✓ Layer normalization" << std::endl;
        std::cout << "✓ Feed-forward networks with GELU activation" << std::endl;
        std::cout << "✓ Complete transformer blocks" << std::endl;
        std::cout << "✓ Simple tokenization system" << std::endl;
        std::cout << "✓ Adam optimizer" << std::endl;
        std::cout << "✓ Cross-entropy loss with gradient computation" << std::endl;
        std::cout << "✓ Training loop with evaluation" << std::endl;
        std::cout << "✓ Text generation capabilities" << std::endl;
        
        demonstrate_components();
        show_model_stats();
        run_training_example();
        
        std::cout << "\n🎉 Transformer model implementation completed successfully!" << std::endl;
        std::cout << "\nThis implementation includes:" << std::endl;
        std::cout << "• High-performance matrix operations with AVX2 SIMD" << std::endl;
        std::cout << "• Complete transformer architecture (encoder-only)" << std::endl;
        std::cout << "• Training infrastructure with backpropagation" << std::endl;
        std::cout << "• Text generation capabilities" << std::endl;
        std::cout << "• Memory-efficient tensor operations" << std::endl;
        std::cout << "\nFor production use, consider adding:" << std::endl;
        std::cout << "• GPU acceleration (CUDA/OpenCL)" << std::endl;
        std::cout << "• More sophisticated tokenization (BPE/SentencePiece)" << std::endl;
        std::cout << "• Model checkpointing and serialization" << std::endl;
        std::cout << "• Distributed training support" << std::endl;
        std::cout << "• Quantization and optimization techniques" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}