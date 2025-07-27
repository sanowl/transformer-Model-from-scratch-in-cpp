#pragma once

#include "matrix.h"
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

// Forward declaration for gradient computation
class GradientContext;

class Tensor {
private:
    std::vector<size_t> shape_;
    std::unique_ptr<Matrix> data_;
    bool requires_grad_;
    std::unique_ptr<Tensor> grad_;
    
    // Automatic differentiation support
    std::function<void()> backward_fn_;
    std::vector<std::weak_ptr<Tensor>> parents_;
    mutable std::shared_ptr<Tensor> self_ptr_;
    bool is_leaf_;
    size_t version_;

public:
    Tensor(const std::vector<size_t>& shape, bool requires_grad = false);
    Tensor(const Matrix& matrix, bool requires_grad = false);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    const std::vector<size_t>& shape() const { return shape_; }
    size_t dim(size_t index) const { return shape_[index]; }
    size_t ndim() const { return shape_.size(); }
    size_t numel() const;
    
    Matrix& matrix() { return *data_; }
    const Matrix& matrix() const { return *data_; }
    
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    
    Tensor& grad() { return *grad_; }
    const Tensor& grad() const { return *grad_; }
    bool has_grad() const { return grad_ != nullptr; }
    void zero_grad();
    void init_grad();
    
    float& operator[](const std::vector<size_t>& indices);
    const float& operator[](const std::vector<size_t>& indices) const;
    
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(float scalar) const;
    
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(float scalar);
    
    Tensor matmul(const Tensor& other) const;
    Tensor transpose(int dim1 = -2, int dim2 = -1) const;
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor view(const std::vector<size_t>& new_shape) const;
    Tensor permute(const std::vector<size_t>& dims) const;
    
    Tensor softmax(int dim = -1) const;
    Tensor relu() const;
    Tensor gelu() const;
    Tensor tanh() const;
    Tensor sigmoid() const;
    
    Tensor sum(int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;
    Tensor var(int dim = -1, bool keepdim = false) const;
    
    void fill_(float value);
    void zero_();
    void normal_(float mean = 0.0f, float std = 1.0f);
    void uniform_(float min = 0.0f, float max = 1.0f);
    void xavier_uniform_(size_t fan_in, size_t fan_out);
    void kaiming_uniform_(size_t fan_in);
    
    void print() const;
    
    // Automatic differentiation methods
    void backward(const Tensor* grad_output = nullptr);
    void set_backward_fn(std::function<void()> fn) { backward_fn_ = fn; }
    void add_parent(std::weak_ptr<Tensor> parent) { parents_.push_back(parent); }
    bool is_leaf() const { return is_leaf_; }
    void set_leaf(bool leaf) { is_leaf_ = leaf; }
    size_t version() const { return version_; }
    void increment_version() { version_++; }
    
    // Smart pointer support
    // Removed shared_from_this to avoid complexity
    void set_self_ptr(std::shared_ptr<Tensor> ptr) const { self_ptr_ = ptr; }
    
public:
    size_t get_flat_index(const std::vector<size_t>& indices) const;
    void validate_shape_compatibility(const Tensor& other) const;
    
private:
    std::vector<size_t> broadcast_shapes(const std::vector<size_t>& shape1, 
                                        const std::vector<size_t>& shape2) const;
    void accumulate_grad(const Tensor& grad);
    void check_grad_compatibility(const Tensor& grad) const;
};

// Factory functions for automatic differentiation
std::shared_ptr<Tensor> make_tensor(const std::vector<size_t>& shape, bool requires_grad = false);
std::shared_ptr<Tensor> make_tensor(const Matrix& matrix, bool requires_grad = false);

// Operations that support automatic differentiation
std::shared_ptr<Tensor> add_op(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> sub_op(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> mul_op(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> matmul_op(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> softmax_op(std::shared_ptr<Tensor> a, int dim = -1);
std::shared_ptr<Tensor> relu_op(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> gelu_op(std::shared_ptr<Tensor> a);

// Gradient context for managing computational graph
class GradientContext {
public:
    static GradientContext& instance();
    void enable_grad(bool enable) { grad_enabled_ = enable; }
    bool is_grad_enabled() const { return grad_enabled_; }
    
private:
    bool grad_enabled_ = true;
};

Tensor operator*(float scalar, const Tensor& tensor);

// Exception classes for better error handling
class TensorShapeError : public std::runtime_error {
public:
    explicit TensorShapeError(const std::string& msg) : std::runtime_error("Tensor shape error: " + msg) {}
};

class TensorGradientError : public std::runtime_error {
public:
    explicit TensorGradientError(const std::string& msg) : std::runtime_error("Tensor gradient error: " + msg) {}
};

class TensorMemoryError : public std::runtime_error {
public:
    explicit TensorMemoryError(const std::string& msg) : std::runtime_error("Tensor memory error: " + msg) {}
};