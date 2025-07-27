#include "matrix.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

Matrix::Matrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols), data(rows * cols, 0.0f) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
}

Matrix::Matrix(size_t rows, size_t cols, float value)
    : rows_(rows), cols_(cols), data(rows * cols, value) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> init) {
    rows_ = init.size();
    if (rows_ == 0) {
        throw std::invalid_argument("Matrix must have at least one row");
    }
    
    cols_ = init.begin()->size();
    if (cols_ == 0) {
        throw std::invalid_argument("Matrix must have at least one column");
    }
    
    data.reserve(rows_ * cols_);
    for (const auto& row : init) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        data.insert(data.end(), row.begin(), row.end());
    }
}

Matrix::Matrix(const Matrix& other) 
    : data(other.data), rows_(other.rows_), cols_(other.cols_) {}

Matrix::Matrix(Matrix&& other) noexcept
    : data(std::move(other.data)), rows_(other.rows_), cols_(other.cols_) {
    other.rows_ = 0;
    other.cols_ = 0;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        data = other.data;
        rows_ = other.rows_;
        cols_ = other.cols_;
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        data = std::move(other.data);
        rows_ = other.rows_;
        cols_ = other.cols_;
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

float& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[row * cols_ + col];
}

const float& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[row * cols_ + col];
}

void Matrix::validate_dimensions(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
}

Matrix Matrix::operator+(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_);
    
    const size_t simd_size = 8;
    const size_t simd_end = (data.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        __m256 b = _mm256_loadu_ps(&other.data[i]);
        __m256 sum = _mm256_add_ps(a, b);
        _mm256_storeu_ps(&result.data[i], sum);
    }
    
    for (size_t i = simd_end; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_);
    
    const size_t simd_size = 8;
    const size_t simd_end = (data.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        __m256 b = _mm256_loadu_ps(&other.data[i]);
        __m256 diff = _mm256_sub_ps(a, b);
        _mm256_storeu_ps(&result.data[i], diff);
    }
    
    for (size_t i = simd_end; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    
    return result;
}

void Matrix::matmul_simd(const float* a, const float* b, float* c,
                        size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; j += 8) {
            __m256 sum = _mm256_setzero_ps();
            
            for (size_t l = 0; l < k; ++l) {
                __m256 a_vec = _mm256_broadcast_ss(&a[i * k + l]);
                __m256 b_vec = _mm256_loadu_ps(&b[l * n + j]);
                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
            }
            
            if (j + 8 <= n) {
                _mm256_storeu_ps(&c[i * n + j], sum);
            } else {
                alignas(32) float temp[8];
                _mm256_store_ps(temp, sum);
                for (size_t idx = 0; idx < n - j; ++idx) {
                    c[i * n + j + idx] = temp[idx];
                }
            }
        }
    }
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows_, other.cols_);
    matmul_simd(data.data(), other.data.data(), result.data.data(),
               rows_, other.cols_, cols_);
    
    return result;
}

Matrix Matrix::operator*(float scalar) const {
    Matrix result(rows_, cols_);
    
    const size_t simd_size = 8;
    const size_t simd_end = (data.size() / simd_size) * simd_size;
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        __m256 product = _mm256_mul_ps(a, scalar_vec);
        _mm256_storeu_ps(&result.data[i], product);
    }
    
    for (size_t i = simd_end; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    validate_dimensions(other);
    
    const size_t simd_size = 8;
    const size_t simd_end = (data.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        __m256 b = _mm256_loadu_ps(&other.data[i]);
        __m256 sum = _mm256_add_ps(a, b);
        _mm256_storeu_ps(&data[i], sum);
    }
    
    for (size_t i = simd_end; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    validate_dimensions(other);
    
    const size_t simd_size = 8;
    const size_t simd_end = (data.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        __m256 b = _mm256_loadu_ps(&other.data[i]);
        __m256 diff = _mm256_sub_ps(a, b);
        _mm256_storeu_ps(&data[i], diff);
    }
    
    for (size_t i = simd_end; i < data.size(); ++i) {
        data[i] -= other.data[i];
    }
    
    return *this;
}

Matrix& Matrix::operator*=(float scalar) {
    const size_t simd_size = 8;
    const size_t simd_end = (data.size() / simd_size) * simd_size;
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        __m256 product = _mm256_mul_ps(a, scalar_vec);
        _mm256_storeu_ps(&data[i], product);
    }
    
    for (size_t i = simd_end; i < data.size(); ++i) {
        data[i] *= scalar;
    }
    
    return *this;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    
    return result;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_);
    
    const size_t simd_size = 8;
    const size_t simd_end = (data.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        __m256 b = _mm256_loadu_ps(&other.data[i]);
        __m256 product = _mm256_mul_ps(a, b);
        _mm256_storeu_ps(&result.data[i], product);
    }
    
    for (size_t i = simd_end; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    
    return result;
}

void Matrix::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

void Matrix::zero() {
    fill(0.0f);
}

void Matrix::randomize(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    for (float& val : data) {
        val = dis(gen);
    }
}

void Matrix::xavier_init(size_t fan_in, size_t fan_out) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    randomize(-limit, limit);
}

void Matrix::he_init(size_t fan_in) {
    float std_dev = std::sqrt(2.0f / fan_in);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, std_dev);
    
    for (float& val : data) {
        val = dis(gen);
    }
}

Matrix Matrix::softmax(int axis) const {
    Matrix result(rows_, cols_);
    
    if (axis == -1 || axis == 1) {
        for (size_t i = 0; i < rows_; ++i) {
            float max_val = *std::max_element(data.begin() + i * cols_, 
                                            data.begin() + (i + 1) * cols_);
            
            float sum = 0.0f;
            for (size_t j = 0; j < cols_; ++j) {
                float exp_val = std::exp((*this)(i, j) - max_val);
                result(i, j) = exp_val;
                sum += exp_val;
            }
            
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) /= sum;
            }
        }
    } else {
        for (size_t j = 0; j < cols_; ++j) {
            float max_val = (*this)(0, j);
            for (size_t i = 1; i < rows_; ++i) {
                max_val = std::max(max_val, (*this)(i, j));
            }
            
            float sum = 0.0f;
            for (size_t i = 0; i < rows_; ++i) {
                float exp_val = std::exp((*this)(i, j) - max_val);
                result(i, j) = exp_val;
                sum += exp_val;
            }
            
            for (size_t i = 0; i < rows_; ++i) {
                result(i, j) /= sum;
            }
        }
    }
    
    return result;
}

Matrix Matrix::relu() const {
    Matrix result(rows_, cols_);
    
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::max(0.0f, data[i]);
    }
    
    return result;
}

Matrix Matrix::gelu() const {
    Matrix result(rows_, cols_);
    
    for (size_t i = 0; i < data.size(); ++i) {
        float x = data[i];
        result.data[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * 
                                    (x + 0.044715f * x * x * x)));
    }
    
    return result;
}

Matrix Matrix::tanh() const {
    Matrix result(rows_, cols_);
    
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::tanh(data[i]);
    }
    
    return result;
}

float Matrix::sum() const {
    float total = 0.0f;
    
    const size_t simd_size = 8;
    const size_t simd_end = (data.size() / simd_size) * simd_size;
    __m256 sum_vec = _mm256_setzero_ps();
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        sum_vec = _mm256_add_ps(sum_vec, a);
    }
    
    alignas(32) float temp[8];
    _mm256_store_ps(temp, sum_vec);
    for (int i = 0; i < 8; ++i) {
        total += temp[i];
    }
    
    for (size_t i = simd_end; i < data.size(); ++i) {
        total += data[i];
    }
    
    return total;
}

float Matrix::mean() const {
    return sum() / static_cast<float>(data.size());
}

float Matrix::variance() const {
    float mean_val = mean();
    float var_sum = 0.0f;
    
    for (float val : data) {
        float diff = val - mean_val;
        var_sum += diff * diff;
    }
    
    return var_sum / static_cast<float>(data.size());
}

Matrix Matrix::sum_axis(int axis) const {
    if (axis == 0) {
        Matrix result(1, cols_);
        for (size_t j = 0; j < cols_; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < rows_; ++i) {
                sum += (*this)(i, j);
            }
            result(0, j) = sum;
        }
        return result;
    } else {
        Matrix result(rows_, 1);
        for (size_t i = 0; i < rows_; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < cols_; ++j) {
                sum += (*this)(i, j);
            }
            result(i, 0) = sum;
        }
        return result;
    }
}

Matrix Matrix::mean_axis(int axis) const {
    if (axis == 0) {
        return sum_axis(axis) * (1.0f / static_cast<float>(rows_));
    } else {
        return sum_axis(axis) * (1.0f / static_cast<float>(cols_));
    }
}

void Matrix::print() const {
    std::cout << "Matrix " << rows_ << "x" << cols_ << ":\n";
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            std::cout << std::setw(10) << std::setprecision(6) << (*this)(i, j);
        }
        std::cout << "\n";
    }
}

Matrix operator*(float scalar, const Matrix& matrix) {
    return matrix * scalar;
}