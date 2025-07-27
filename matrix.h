#pragma once

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <random>
#include <immintrin.h>
#include <algorithm>
#include <memory>

class Matrix {
private:
    std::vector<float> data;
    size_t rows_;
    size_t cols_;

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
    size_t size() const { return data.size(); }
    
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    
    float* raw_data() { return data.data(); }
    const float* raw_data() const { return data.data(); }
    
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
    static void matmul_simd(const float* a, const float* b, float* c, 
                           size_t m, size_t n, size_t k);
};

Matrix operator*(float scalar, const Matrix& matrix);