#pragma once

#include "matrix.h"
#include <vector>
#include <memory>

class Tensor {
private:
    std::vector<size_t> shape_;
    std::unique_ptr<Matrix> data_;
    bool requires_grad_;
    std::unique_ptr<Tensor> grad_;

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
    
private:
    size_t get_flat_index(const std::vector<size_t>& indices) const;
    void validate_shape_compatibility(const Tensor& other) const;
    std::vector<size_t> broadcast_shapes(const std::vector<size_t>& shape1, 
                                        const std::vector<size_t>& shape2) const;
};

Tensor operator*(float scalar, const Tensor& tensor);