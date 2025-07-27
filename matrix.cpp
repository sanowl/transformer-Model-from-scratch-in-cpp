#include "matrix.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <cstdlib>

#ifdef _WIN32
    #include <malloc.h>
#else
    #include <cstdlib>
#endif
#include <thread>
#include <mutex>

Matrix::Matrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {
    if (rows == 0 || cols == 0) {
        throw MatrixDimensionError("Matrix dimensions must be positive");
    }
    
    // Ensure proper memory alignment for SIMD operations
    try {
        data_.reserve(rows * cols);
    } catch (const std::bad_alloc& e) {
        throw MatrixMemoryError("Failed to allocate memory for matrix: " + std::string(e.what()));
    }
}

Matrix::Matrix(size_t rows, size_t cols, float value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {
    if (rows == 0 || cols == 0) {
        throw MatrixDimensionError("Matrix dimensions must be positive");
    }
    
    try {
        data_.reserve(rows * cols);
    } catch (const std::bad_alloc& e) {
        throw MatrixMemoryError("Failed to allocate memory for matrix: " + std::string(e.what()));
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> init) {
    rows_ = init.size();
    if (rows_ == 0) {
        throw MatrixDimensionError("Matrix must have at least one row");
    }
    
    cols_ = init.begin()->size();
    if (cols_ == 0) {
        throw MatrixDimensionError("Matrix must have at least one column");
    }
    
    try {
        data_.reserve(rows_ * cols_);
        for (const auto& row : init) {
            if (row.size() != cols_) {
                throw MatrixDimensionError("All rows must have the same number of columns");
            }
            data_.insert(data_.end(), row.begin(), row.end());
        }
    } catch (const std::bad_alloc& e) {
        throw MatrixMemoryError("Failed to allocate memory for matrix initialization: " + std::string(e.what()));
    }
}

Matrix::Matrix(const Matrix& other) 
    : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {
    try {
        data_.reserve(rows_ * cols_);
    } catch (const std::bad_alloc& e) {
        throw MatrixMemoryError("Failed to copy matrix: " + std::string(e.what()));
    }
}

Matrix::Matrix(Matrix&& other) noexcept
    : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_) {
    other.rows_ = 0;
    other.cols_ = 0;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        try {
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
        } catch (const std::bad_alloc& e) {
            throw MatrixMemoryError("Failed to assign matrix: " + std::string(e.what()));
        }
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        rows_ = other.rows_;
        cols_ = other.cols_;
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

void Matrix::validate_index(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw MatrixIndexError("Index (" + std::to_string(row) + ", " + std::to_string(col) + 
                              ") out of bounds for matrix of size (" + std::to_string(rows_) + 
                              ", " + std::to_string(cols_) + ")");
    }
}

float& Matrix::operator()(size_t row, size_t col) {
    validate_index(row, col);
    return data_[row * cols_ + col];
}

const float& Matrix::operator()(size_t row, size_t col) const {
    validate_index(row, col);
    return data_[row * cols_ + col];
}

void Matrix::validate_dimensions(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw MatrixDimensionError("Matrix dimensions must match. This: (" + 
                                 std::to_string(rows_) + ", " + std::to_string(cols_) + 
                                 "), Other: (" + std::to_string(other.rows_) + ", " + 
                                 std::to_string(other.cols_) + ")");
    }
}

void Matrix::ensure_capacity(size_t min_size) {
    try {
        if (data_.capacity() < min_size) {
            data_.reserve(min_size);
        }
    } catch (const std::bad_alloc& e) {
        throw MatrixMemoryError("Failed to ensure matrix capacity: " + std::string(e.what()));
    }
}

bool Matrix::is_simd_aligned() const {
    return reinterpret_cast<uintptr_t>(data_.data()) % SIMD_ALIGNMENT == 0;
}

Matrix Matrix::operator+(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_);
    
#ifdef USE_X86_SIMD
#ifdef USE_X86_SIMD
    const size_t simd_size = 8;
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
#else
    const size_t simd_size = 1;
#endif
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data_[i]);
        __m256 b = _mm256_loadu_ps(&other.data_[i]);
        __m256 sum = _mm256_add_ps(a, b);
        _mm256_storeu_ps(&result.data_[i], sum);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        float32x4_t a = vld1q_f32(&data_[i]);
        float32x4_t b = vld1q_f32(&other.data_[i]);
        float32x4_t sum = vaddq_f32(a, b);
        vst1q_f32(&result.data_[i], sum);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
#else
    // Portable fallback
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
#endif
    
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_);
    
#ifdef USE_X86_SIMD
#ifdef USE_X86_SIMD
    const size_t simd_size = 8;
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
#else
    const size_t simd_size = 1;
#endif
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data_[i]);
        __m256 b = _mm256_loadu_ps(&other.data_[i]);
        __m256 diff = _mm256_sub_ps(a, b);
        _mm256_storeu_ps(&result.data_[i], diff);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        float32x4_t a = vld1q_f32(&data_[i]);
        float32x4_t b = vld1q_f32(&other.data_[i]);
        float32x4_t diff = vsubq_f32(a, b);
        vst1q_f32(&result.data_[i], diff);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
#else
    // Portable fallback
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
#endif
    
    return result;
}

void Matrix::matmul_simd_optimized(const float* a, const float* b, float* c,
                                   size_t m, size_t n, size_t k) {
#ifdef USE_X86_SIMD
    // Optimized matrix multiplication with better cache utilization
    const size_t block_size = 64; // Cache-friendly block size
    
    for (size_t ii = 0; ii < m; ii += block_size) {
        for (size_t jj = 0; jj < n; jj += block_size) {
            for (size_t kk = 0; kk < k; kk += block_size) {
                // Process blocks
                size_t i_end = std::min(ii + block_size, m);
                size_t j_end = std::min(jj + block_size, n);
                size_t k_end = std::min(kk + block_size, k);
                
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; j += SIMD_WIDTH) {
                        __m256 sum = _mm256_setzero_ps();
                        
                        for (size_t l = kk; l < k_end; ++l) {
                            __m256 a_vec = _mm256_broadcast_ss(&a[i * k + l]);
                            
                            if (j + SIMD_WIDTH <= j_end) {
                                __m256 b_vec = _mm256_loadu_ps(&b[l * n + j]);
                                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                            } else {
                                // Handle remaining elements
                                alignas(32) float b_temp[8] = {0};
                                for (size_t idx = 0; idx < j_end - j; ++idx) {
                                    b_temp[idx] = b[l * n + j + idx];
                                }
                                __m256 b_vec = _mm256_load_ps(b_temp);
                                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                            }
                        }
                        
                        if (j + SIMD_WIDTH <= j_end) {
                            _mm256_storeu_ps(&c[i * n + j], sum);
                        } else {
                            alignas(32) float temp[8];
                            _mm256_store_ps(temp, sum);
                            for (size_t idx = 0; idx < j_end - j; ++idx) {
                                c[i * n + j + idx] = temp[idx];
                            }
                        }
                    }
                }
            }
        }
    }
#else
    // Fallback to regular matmul_simd for non-x86 architectures
    matmul_simd(a, b, c, m, n, k);
#endif
}

void Matrix::matmul_simd(const float* a, const float* b, float* c,
                        size_t m, size_t n, size_t k) {
#ifdef USE_X86_SIMD
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; j += 8) {
            __m256 sum = _mm256_setzero_ps();
            
            for (size_t l = 0; l < k; ++l) {
                __m256 a_vec = _mm256_broadcast_ss(&a[i * k + l]);
                if (j + 8 <= n) {
                    __m256 b_vec = _mm256_loadu_ps(&b[l * n + j]);
                    sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                }
            }
            
            if (j + 8 <= n) {
                _mm256_storeu_ps(&c[i * n + j], sum);
            } else {
                alignas(32) float temp[8];
                _mm256_store_ps(temp, sum);
                for (size_t idx = 0; idx < n - j && j + idx < n; ++idx) {
                    c[i * n + j + idx] = temp[idx];
                }
            }
        }
    }
#elif defined(USE_ARM_NEON)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; j += 4) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            
            for (size_t l = 0; l < k; ++l) {
                float32x4_t a_vec = vdupq_n_f32(a[i * k + l]);
                if (j + 4 <= n) {
                    float32x4_t b_vec = vld1q_f32(&b[l * n + j]);
                    sum = vfmaq_f32(sum, a_vec, b_vec);
                }
            }
            
            if (j + 4 <= n) {
                vst1q_f32(&c[i * n + j], sum);
            } else {
                float temp[4];
                vst1q_f32(temp, sum);
                for (size_t idx = 0; idx < n - j && j + idx < n; ++idx) {
                    c[i * n + j + idx] = temp[idx];
                }
            }
        }
    }
#else
    // Portable fallback - standard matrix multiplication
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
#endif
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows_, other.cols_);
    matmul_simd(data_.data(), other.data_.data(), result.data_.data(),
               rows_, other.cols_, cols_);
    
    return result;
}

Matrix Matrix::operator*(float scalar) const {
    Matrix result(rows_, cols_);
    
#ifdef USE_X86_SIMD
#ifdef USE_X86_SIMD
    const size_t simd_size = 8;
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
#else
    const size_t simd_size = 1;
#endif
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data_[i]);
        __m256 product = _mm256_mul_ps(a, scalar_vec);
        _mm256_storeu_ps(&result.data_[i], product);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    float32x4_t scalar_vec = vdupq_n_f32(scalar);
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        float32x4_t a = vld1q_f32(&data_[i]);
        float32x4_t product = vmulq_f32(a, scalar_vec);
        vst1q_f32(&result.data_[i], product);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
#else
    // Portable fallback
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
#endif
    
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    validate_dimensions(other);
    
#ifdef USE_X86_SIMD
#ifdef USE_X86_SIMD
    const size_t simd_size = 8;
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
#else
    const size_t simd_size = 1;
#endif
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data_[i]);
        __m256 b = _mm256_loadu_ps(&other.data_[i]);
        __m256 sum = _mm256_add_ps(a, b);
        _mm256_storeu_ps(&data_[i], sum);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        float32x4_t a = vld1q_f32(&data_[i]);
        float32x4_t b = vld1q_f32(&other.data_[i]);
        float32x4_t sum = vaddq_f32(a, b);
        vst1q_f32(&data_[i], sum);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
#else
    // Portable fallback
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
#endif
    
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    validate_dimensions(other);
    
#ifdef USE_X86_SIMD
#ifdef USE_X86_SIMD
    const size_t simd_size = 8;
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
#else
    const size_t simd_size = 1;
#endif
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data_[i]);
        __m256 b = _mm256_loadu_ps(&other.data_[i]);
        __m256 diff = _mm256_sub_ps(a, b);
        _mm256_storeu_ps(&data_[i], diff);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        float32x4_t a = vld1q_f32(&data_[i]);
        float32x4_t b = vld1q_f32(&other.data_[i]);
        float32x4_t diff = vsubq_f32(a, b);
        vst1q_f32(&data_[i], diff);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
#else
    // Portable fallback
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
#endif
    
    return *this;
}

Matrix& Matrix::operator*=(float scalar) {
#ifdef USE_X86_SIMD
#ifdef USE_X86_SIMD
    const size_t simd_size = 8;
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
#else
    const size_t simd_size = 1;
#endif
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data_[i]);
        __m256 product = _mm256_mul_ps(a, scalar_vec);
        _mm256_storeu_ps(&data_[i], product);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        data_[i] *= scalar;
    }
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    float32x4_t scalar_vec = vdupq_n_f32(scalar);
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        float32x4_t a = vld1q_f32(&data_[i]);
        float32x4_t product = vmulq_f32(a, scalar_vec);
        vst1q_f32(&data_[i], product);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        data_[i] *= scalar;
    }
#else
    // Portable fallback
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] *= scalar;
    }
#endif
    
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
    
#ifdef USE_X86_SIMD
#ifdef USE_X86_SIMD
    const size_t simd_size = 8;
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
#else
    const size_t simd_size = 1;
#endif
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data_[i]);
        __m256 b = _mm256_loadu_ps(&other.data_[i]);
        __m256 product = _mm256_mul_ps(a, b);
        _mm256_storeu_ps(&result.data_[i], product);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        float32x4_t a = vld1q_f32(&data_[i]);
        float32x4_t b = vld1q_f32(&other.data_[i]);
        float32x4_t product = vmulq_f32(a, b);
        vst1q_f32(&result.data_[i], product);
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
#else
    // Portable fallback
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
#endif
    
    return result;
}

void Matrix::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Matrix::zero() {
    fill(0.0f);
}

void Matrix::randomize(float min, float max) {
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    for (float& val : data_) {
        val = dis(gen);
    }
}

void Matrix::xavier_init(size_t fan_in, size_t fan_out) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    randomize(-limit, limit);
}

void Matrix::he_init(size_t fan_in) {
    float std_dev = std::sqrt(2.0f / fan_in);
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, std_dev);
    
    for (float& val : data_) {
        val = dis(gen);
    }
}

Matrix Matrix::softmax(int axis) const {
    Matrix result(rows_, cols_);
    
    if (axis == -1 || axis == 1) {
        for (size_t i = 0; i < rows_; ++i) {
            float max_val = *std::max_element(data_.begin() + i * cols_, 
                                            data_.begin() + (i + 1) * cols_);
            
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
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::max(0.0f, data_[i]);
    }
    
    return result;
}

Matrix Matrix::gelu() const {
    Matrix result(rows_, cols_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        float x = data_[i];
        result.data_[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * 
                                    (x + 0.044715f * x * x * x)));
    }
    
    return result;
}

Matrix Matrix::tanh() const {
    Matrix result(rows_, cols_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::tanh(data_[i]);
    }
    
    return result;
}

float Matrix::sum() const {
    float total = 0.0f;
    
#ifdef USE_X86_SIMD
#ifdef USE_X86_SIMD
    const size_t simd_size = 8;
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
#else
    const size_t simd_size = 1;
#endif
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    __m256 sum_vec = _mm256_setzero_ps();
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 a = _mm256_loadu_ps(&data_[i]);
        sum_vec = _mm256_add_ps(sum_vec, a);
    }
    
    alignas(32) float temp[8];
    _mm256_store_ps(temp, sum_vec);
    for (int i = 0; i < 8; ++i) {
        total += temp[i];
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        total += data_[i];
    }
#elif defined(USE_ARM_NEON)
    const size_t simd_size = 4;
    const size_t simd_end = (data_.size() / simd_size) * simd_size;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        float32x4_t a = vld1q_f32(&data_[i]);
        sum_vec = vaddq_f32(sum_vec, a);
    }
    
    float temp[4];
    vst1q_f32(temp, sum_vec);
    for (int i = 0; i < 4; ++i) {
        total += temp[i];
    }
    
    for (size_t i = simd_end; i < data_.size(); ++i) {
        total += data_[i];
    }
#else
    // Portable fallback
    for (size_t i = 0; i < data_.size(); ++i) {
        total += data_[i];
    }
#endif
    
    return total;
}

float Matrix::mean() const {
    return sum() / static_cast<float>(data_.size());
}

float Matrix::variance() const {
    float mean_val = mean();
    float var_sum = 0.0f;
    
    for (float val : data_) {
        float diff = val - mean_val;
        var_sum += diff * diff;
    }
    
    return var_sum / static_cast<float>(data_.size());
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

// AlignedMemory implementation
AlignedMemory::AlignedMemory(size_t size, size_t alignment) 
    : size_(size), alignment_(alignment) {
    if (size == 0) {
        throw MatrixMemoryError("Cannot allocate zero-sized memory");
    }
    
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw MatrixMemoryError("Alignment must be a power of two");
    }
    
#ifdef _WIN32
    data_ = static_cast<float*>(_aligned_malloc(size * sizeof(float), alignment));
    if (!data_) {
        throw MatrixMemoryError("Failed to allocate aligned memory");
    }
#else
    if (posix_memalign(reinterpret_cast<void**>(&data_), alignment, size * sizeof(float)) != 0) {
        throw MatrixMemoryError("Failed to allocate aligned memory");
    }
#endif
}

AlignedMemory::~AlignedMemory() {
    if (data_) {
#ifdef _WIN32
        _aligned_free(data_);
#else
        free(data_);
#endif
        data_ = nullptr;
    }
}

AlignedMemory::AlignedMemory(AlignedMemory&& other) noexcept
    : data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

AlignedMemory& AlignedMemory::operator=(AlignedMemory&& other) noexcept {
    if (this != &other) {
        // Clean up current memory
        if (data_) {
#ifdef _WIN32
            _aligned_free(data_);
#else
            free(data_);
#endif
        }
        
        // Take ownership of other's memory
        data_ = other.data_;
        size_ = other.size_;
        alignment_ = other.alignment_;
        
        // Reset other
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}