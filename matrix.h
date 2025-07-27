#pragma once

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <memory>
#include <cmath>
#include <cstring>
#include <cassert>

// Cross-platform SIMD detection and includes
#if defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || defined(_M_AMD64)
    #include <immintrin.h>
    #define USE_X86_SIMD
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || (defined(__ARM_ARCH) && (__ARM_ARCH >= 7)) || defined(__aarch64__)
    #include <arm_neon.h>
    #define USE_ARM_NEON
#endif

class Matrix {
private:
    std::vector<float> data_;
    size_t rows_;
    size_t cols_;
    
    // Memory alignment for SIMD operations
#ifdef USE_X86_SIMD
    static constexpr size_t SIMD_ALIGNMENT = 32;
    static constexpr size_t SIMD_WIDTH = 8; // AVX2 processes 8 floats at once
#elif defined(USE_ARM_NEON)
    static constexpr size_t SIMD_ALIGNMENT = 16;
    static constexpr size_t SIMD_WIDTH = 4; // NEON processes 4 floats at once
#else
    static constexpr size_t SIMD_ALIGNMENT = 16;
    static constexpr size_t SIMD_WIDTH = 1; // No SIMD, process 1 float at once
#endif

public:
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, float value);
    Matrix(std::initializer_list<std::initializer_list<float>> init);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }
    
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    
    float* raw_data() { return data_.data(); }
    const float* raw_data() const { return data_.data(); }
    
    // Safe data access with bounds checking
    // Note: std::span not available in C++17, using raw pointers with size
    float* data_ptr() { return data_.data(); }
    const float* data_ptr() const { return data_.data(); }
    
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(float scalar) const;
    
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(float scalar);
    
    Matrix transpose() const;
    Matrix hadamard(const Matrix& other) const;
    
    void fill(float value);
    void zero();
    void randomize(float min = -1.0f, float max = 1.0f);
    void xavier_init(size_t fan_in, size_t fan_out);
    void he_init(size_t fan_in);
    
    Matrix softmax(int axis = -1) const;
    Matrix relu() const;
    Matrix gelu() const;
    Matrix tanh() const;
    
    float sum() const;
    float mean() const;
    float variance() const;
    Matrix sum_axis(int axis) const;
    Matrix mean_axis(int axis) const;
    
    void print() const;
    
private:
    void validate_dimensions(const Matrix& other) const;
    void validate_index(size_t row, size_t col) const;
    static void matmul_simd(const float* a, const float* b, float* c, 
                           size_t m, size_t n, size_t k);
    static void matmul_simd_optimized(const float* a, const float* b, float* c,
                                     size_t m, size_t n, size_t k);
    void ensure_capacity(size_t min_size);
    bool is_simd_aligned() const;
};

// RAII wrapper for aligned memory
class AlignedMemory {
public:
    explicit AlignedMemory(size_t size, size_t alignment = 32);
    ~AlignedMemory();
    
    // Non-copyable but movable
    AlignedMemory(const AlignedMemory&) = delete;
    AlignedMemory& operator=(const AlignedMemory&) = delete;
    AlignedMemory(AlignedMemory&& other) noexcept;
    AlignedMemory& operator=(AlignedMemory&& other) noexcept;
    
    float* data() { return data_; }
    const float* data() const { return data_; }
    size_t size() const { return size_; }
    
private:
    float* data_;
    size_t size_;
    size_t alignment_;
};

Matrix operator*(float scalar, const Matrix& matrix);

// Exception classes for better error handling
class MatrixDimensionError : public std::runtime_error {
public:
    explicit MatrixDimensionError(const std::string& msg) 
        : std::runtime_error("Matrix dimension error: " + msg) {}
};

class MatrixMemoryError : public std::runtime_error {
public:
    explicit MatrixMemoryError(const std::string& msg) 
        : std::runtime_error("Matrix memory error: " + msg) {}
};

class MatrixIndexError : public std::runtime_error {
public:
    explicit MatrixIndexError(const std::string& msg) 
        : std::runtime_error("Matrix index error: " + msg) {}
};