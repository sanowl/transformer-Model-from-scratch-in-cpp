#include "tensor.h"
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <queue>

Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
    : shape_(shape), requires_grad_(requires_grad), is_leaf_(true), version_(0) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty");
    }
    
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    
    if (shape.size() == 1) {
        data_ = std::make_unique<Matrix>(1, shape[0]);
    } else if (shape.size() == 2) {
        data_ = std::make_unique<Matrix>(shape[0], shape[1]);
    } else {
        size_t rows = std::accumulate(shape.begin(), shape.end() - 1, 1ULL, std::multiplies<size_t>());
        data_ = std::make_unique<Matrix>(rows, shape.back());
    }
}

Tensor::Tensor(const Matrix& matrix, bool requires_grad)
    : requires_grad_(requires_grad), is_leaf_(true), version_(0) {
    shape_ = {matrix.rows(), matrix.cols()};
    data_ = std::make_unique<Matrix>(matrix);
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), requires_grad_(other.requires_grad_), 
      is_leaf_(other.is_leaf_), version_(0) {
    data_ = std::make_unique<Matrix>(*other.data_);
    if (other.grad_) {
        grad_ = std::make_unique<Tensor>(*other.grad_);
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), requires_grad_(other.requires_grad_),
      data_(std::move(other.data_)), grad_(std::move(other.grad_)),
      backward_fn_(std::move(other.backward_fn_)), parents_(std::move(other.parents_)),
      is_leaf_(other.is_leaf_), version_(other.version_) {
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        requires_grad_ = other.requires_grad_;
        data_ = std::make_unique<Matrix>(*other.data_);
        if (other.grad_) {
            grad_ = std::make_unique<Tensor>(*other.grad_);
        } else {
            grad_.reset();
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        requires_grad_ = other.requires_grad_;
        data_ = std::move(other.data_);
        grad_ = std::move(other.grad_);
    }
    return *this;
}

size_t Tensor::numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
}

void Tensor::zero_grad() {
    if (grad_) {
        grad_->zero_();
    }
}

void Tensor::init_grad() {
    if (requires_grad_ && !grad_) {
        try {
            grad_ = std::make_unique<Tensor>(shape_, false);
            grad_->zero_();
        } catch (const std::exception& e) {
            throw TensorMemoryError("Failed to initialize gradient tensor: " + std::string(e.what()));
        }
    }
}

void Tensor::check_grad_compatibility(const Tensor& grad) const {
    if (grad.shape() != shape_) {
        throw TensorGradientError("Gradient shape mismatch. Expected shape matches tensor shape");
    }
}

void Tensor::accumulate_grad(const Tensor& grad) {
    check_grad_compatibility(grad);
    
    if (!grad_) {
        init_grad();
    }
    
    *grad_ += grad;
}

// Removed shared_from_this function for simplicity

void Tensor::backward(const Tensor* grad_output) {
    if (!requires_grad_) {
        return;
    }
    
    // Initialize gradient if this is the root of backpropagation
    if (grad_output == nullptr) {
        if (numel() != 1) {
            throw TensorGradientError("Backward can only be called on scalar tensors without grad_output");
        }
        init_grad();
        grad_->fill_(1.0f);
    } else {
        accumulate_grad(*grad_output);
    }
    
    // Execute backward function if available
    if (backward_fn_) {
        backward_fn_();
    }
}

size_t Tensor::get_flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    
    size_t flat_index = 0;
    size_t stride = 1;
    
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        flat_index += indices[i] * stride;
        stride *= shape_[i];
    }
    
    return flat_index;
}

float& Tensor::operator[](const std::vector<size_t>& indices) {
    size_t flat_idx = get_flat_index(indices);
    
    if (shape_.size() <= 2) {
        size_t row = flat_idx / data_->cols();
        size_t col = flat_idx % data_->cols();
        return (*data_)(row, col);
    } else {
        size_t row = flat_idx / shape_.back();
        size_t col = flat_idx % shape_.back();
        return (*data_)(row, col);
    }
}

const float& Tensor::operator[](const std::vector<size_t>& indices) const {
    size_t flat_idx = get_flat_index(indices);
    
    if (shape_.size() <= 2) {
        size_t row = flat_idx / data_->cols();
        size_t col = flat_idx % data_->cols();
        return (*data_)(row, col);
    } else {
        size_t row = flat_idx / shape_.back();
        size_t col = flat_idx % shape_.back();
        return (*data_)(row, col);
    }
}

void Tensor::validate_shape_compatibility(const Tensor& other) const {
    if (shape_ != other.shape_) {
        std::string msg = "Tensor shapes are incompatible. This: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            msg += std::to_string(shape_[i]);
            if (i < shape_.size() - 1) msg += ", ";
        }
        msg += "], Other: [";
        for (size_t i = 0; i < other.shape_.size(); ++i) {
            msg += std::to_string(other.shape_[i]);
            if (i < other.shape_.size() - 1) msg += ", ";
        }
        msg += "]";
        throw TensorShapeError(msg);
    }
}

Tensor Tensor::operator+(const Tensor& other) const {
    validate_shape_compatibility(other);
    Tensor result(shape_, requires_grad_ || other.requires_grad_);
    result.matrix() = matrix() + other.matrix();
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    validate_shape_compatibility(other);
    Tensor result(shape_, requires_grad_ || other.requires_grad_);
    result.matrix() = matrix() - other.matrix();
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    validate_shape_compatibility(other);
    Tensor result(shape_, requires_grad_ || other.requires_grad_);
    result.matrix() = matrix().hadamard(other.matrix());
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_, requires_grad_);
    result.matrix() = matrix() * scalar;
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    validate_shape_compatibility(other);
    matrix() += other.matrix();
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    validate_shape_compatibility(other);
    matrix() -= other.matrix();
    return *this;
}

Tensor& Tensor::operator*=(float scalar) {
    matrix() *= scalar;
    return *this;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }
    
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Incompatible shapes for matrix multiplication");
    }
    
    std::vector<size_t> result_shape = {shape_[0], other.shape_[1]};
    Tensor result(result_shape, requires_grad_ || other.requires_grad_);
    result.matrix() = matrix() * other.matrix();
    
    return result;
}

Tensor Tensor::transpose(int dim1, int dim2) const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Transpose currently only supports 2D tensors");
    }
    
    std::vector<size_t> new_shape = {shape_[1], shape_[0]};
    Tensor result(new_shape, requires_grad_);
    result.matrix() = matrix().transpose();
    
    return result;
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    if (new_numel != numel()) {
        throw std::invalid_argument("Reshape size mismatch");
    }
    
    Tensor result(new_shape, requires_grad_);
    
    for (size_t i = 0; i < numel(); ++i) {
        size_t old_row = i / matrix().cols();
        size_t old_col = i % matrix().cols();
        size_t new_row = i / result.matrix().cols();
        size_t new_col = i % result.matrix().cols();
        result.matrix()(new_row, new_col) = matrix()(old_row, old_col);
    }
    
    return result;
}

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    return reshape(new_shape);
}

Tensor Tensor::softmax(int dim) const {
    Tensor result(shape_, requires_grad_);
    result.matrix() = matrix().softmax(dim);
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(shape_, requires_grad_);
    result.matrix() = matrix().relu();
    return result;
}

Tensor Tensor::gelu() const {
    Tensor result(shape_, requires_grad_);
    result.matrix() = matrix().gelu();
    return result;
}

Tensor Tensor::tanh() const {
    Tensor result(shape_, requires_grad_);
    result.matrix() = matrix().tanh();
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape_, requires_grad_);
    Matrix& result_matrix = result.matrix();
    const Matrix& input_matrix = matrix();
    
    for (size_t i = 0; i < input_matrix.rows(); ++i) {
        for (size_t j = 0; j < input_matrix.cols(); ++j) {
            float x = input_matrix(i, j);
            result_matrix(i, j) = 1.0f / (1.0f + std::exp(-x));
        }
    }
    
    return result;
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    if (dim == -1) {
        std::vector<size_t> result_shape = keepdim ? shape_ : std::vector<size_t>{1};
        if (!keepdim) {
            std::fill(result_shape.begin(), result_shape.end(), 1);
        }
        
        Tensor result(result_shape, requires_grad_);
        result.matrix()(0, 0) = matrix().sum();
        return result;
    } else {
        Matrix sum_result = matrix().sum_axis(dim);
        std::vector<size_t> result_shape;
        
        if (keepdim) {
            result_shape = shape_;
            result_shape[dim] = 1;
        } else {
            for (size_t i = 0; i < shape_.size(); ++i) {
                if (i != static_cast<size_t>(dim)) {
                    result_shape.push_back(shape_[i]);
                }
            }
            if (result_shape.empty()) result_shape.push_back(1);
        }
        
        Tensor result(result_shape, requires_grad_);
        result.matrix() = sum_result;
        return result;
    }
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    Tensor sum_result = sum(dim, keepdim);
    float divisor = (dim == -1) ? static_cast<float>(numel()) : static_cast<float>(shape_[dim]);
    return sum_result * (1.0f / divisor);
}

Tensor Tensor::var(int dim, bool keepdim) const {
    Tensor mean_val = mean(dim, true);
    Tensor centered = *this - mean_val;
    Tensor squared = centered * centered;
    return squared.mean(dim, keepdim);
}

void Tensor::fill_(float value) {
    matrix().fill(value);
}

void Tensor::zero_() {
    matrix().zero();
}

void Tensor::normal_(float mean, float std) {
    matrix().randomize(mean - 3*std, mean + 3*std);
}

void Tensor::uniform_(float min, float max) {
    matrix().randomize(min, max);
}

void Tensor::xavier_uniform_(size_t fan_in, size_t fan_out) {
    matrix().xavier_init(fan_in, fan_out);
}

void Tensor::kaiming_uniform_(size_t fan_in) {
    matrix().he_init(fan_in);
}

void Tensor::print() const {
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "], requires_grad: " << (requires_grad_ ? "true" : "false") << "\n";
    matrix().print();
}

Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

std::vector<size_t> Tensor::broadcast_shapes(const std::vector<size_t>& shape1, 
                                        const std::vector<size_t>& shape2) const {
    // NumPy-style broadcasting rules
    size_t max_dims = std::max(shape1.size(), shape2.size());
    std::vector<size_t> result_shape(max_dims);
    
    // Pad shorter shape with 1s on the left
    std::vector<size_t> padded_shape1(max_dims, 1);
    std::vector<size_t> padded_shape2(max_dims, 1);
    
    // Copy shapes to the right (end) of padded vectors
    std::copy(shape1.rbegin(), shape1.rend(), padded_shape1.rbegin());
    std::copy(shape2.rbegin(), shape2.rend(), padded_shape2.rbegin());
    
    // Apply broadcasting rules
    for (size_t i = 0; i < max_dims; ++i) {
        size_t dim1 = padded_shape1[i];
        size_t dim2 = padded_shape2[i];
        
        if (dim1 == dim2) {
            result_shape[i] = dim1;
        } else if (dim1 == 1) {
            result_shape[i] = dim2;
        } else if (dim2 == 1) {
            result_shape[i] = dim1;
        } else {
            throw TensorShapeError("Incompatible shapes for broadcasting: dimension " + 
                                 std::to_string(i) + " has sizes " + std::to_string(dim1) + 
                                 " and " + std::to_string(dim2));
        }
    }
    
    return result_shape;
}

// Factory functions
std::shared_ptr<Tensor> make_tensor(const std::vector<size_t>& shape, bool requires_grad) {
    auto tensor = std::make_shared<Tensor>(shape, requires_grad);
    tensor->set_self_ptr(tensor);
    return tensor;
}

std::shared_ptr<Tensor> make_tensor(const Matrix& matrix, bool requires_grad) {
    auto tensor = std::make_shared<Tensor>(matrix, requires_grad);
    tensor->set_self_ptr(tensor);
    return tensor;
}

// Automatic differentiation operations
std::shared_ptr<Tensor> add_op(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a || !b) {
        throw TensorMemoryError("Null tensor pointer in add operation");
    }
    
    a->validate_shape_compatibility(*b);
    auto result = make_tensor(a->shape(), a->requires_grad() || b->requires_grad());
    result->matrix() = a->matrix() + b->matrix();
    result->set_leaf(false);
    
    if (GradientContext::instance().is_grad_enabled() && result->requires_grad()) {
        result->add_parent(a);
        result->add_parent(b);
        
        result->set_backward_fn([a, b, result]() {
            if (a->requires_grad()) {
                a->accumulate_grad(*result->grad_);
            }
            if (b->requires_grad()) {
                b->accumulate_grad(*result->grad_);
            }
        });
    }
    
    return result;
}

std::shared_ptr<Tensor> sub_op(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a || !b) {
        throw TensorMemoryError("Null tensor pointer in sub operation");
    }
    
    a->validate_shape_compatibility(*b);
    auto result = make_tensor(a->shape(), a->requires_grad() || b->requires_grad());
    result->matrix() = a->matrix() - b->matrix();
    result->set_leaf(false);
    
    if (GradientContext::instance().is_grad_enabled() && result->requires_grad()) {
        result->add_parent(a);
        result->add_parent(b);
        
        result->set_backward_fn([a, b, result]() {
            if (a->requires_grad()) {
                a->accumulate_grad(*result->grad_);
            }
            if (b->requires_grad()) {
                auto neg_grad = *result->grad_ * (-1.0f);
                b->accumulate_grad(neg_grad);
            }
        });
    }
    
    return result;
}

std::shared_ptr<Tensor> matmul_op(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a || !b) {
        throw TensorMemoryError("Null tensor pointer in matmul operation");
    }
    
    if (a->shape().size() != 2 || b->shape().size() != 2) {
        throw TensorShapeError("Matrix multiplication requires 2D tensors");
    }
    
    if (a->shape()[1] != b->shape()[0]) {
        throw TensorShapeError("Incompatible shapes for matrix multiplication");
    }
    
    std::vector<size_t> result_shape = {a->shape()[0], b->shape()[1]};
    auto result = make_tensor(result_shape, a->requires_grad() || b->requires_grad());
    result->matrix() = a->matrix() * b->matrix();
    result->set_leaf(false);
    
    if (GradientContext::instance().is_grad_enabled() && result->requires_grad()) {
        result->add_parent(a);
        result->add_parent(b);
        
        result->set_backward_fn([a, b, result]() {
            if (a->requires_grad()) {
                auto a_grad = result->grad_->matmul(b->transpose());
                a->accumulate_grad(a_grad);
            }
            if (b->requires_grad()) {
                auto b_grad = a->transpose().matmul(*result->grad_);
                b->accumulate_grad(b_grad);
            }
        });
    }
    
    return result;
}

// Gradient context singleton
GradientContext& GradientContext::instance() {
    static GradientContext ctx;
    return ctx;
}