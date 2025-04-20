#include "engine.hpp"

Value::Value(double d): data{d}, prev{}, grad{0.0} {}

Value::Value(double data, std::vector<std::shared_ptr<Value>> prev): data{data}, prev{prev}, grad{0.0}, _backward{} {}

std::shared_ptr<Value> Value::operator+(std::shared_ptr<Value> other) {
    std::shared_ptr<Value> out = std::shared_ptr<Value>(new Value(this->data + other->data, {shared_from_this(), other}));

    std::function<void(void)> add_backward = [out]() {
        out->prev[0]->grad += out->grad;
        out->prev[1]->grad += out->grad;
    };
    
    out->_backward = add_backward;
    return out;
}

std::shared_ptr<Value> operator+(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
    return a->operator+(b);
}

std::shared_ptr<Value> Value::operator-(std::shared_ptr<Value> other) {
    std::shared_ptr<Value> out = std::shared_ptr<Value>(new Value(this->data - other->data, {shared_from_this(), other}));

    std::function<void(void)> sub_backward = [out]() {
        out->prev[0]->grad += out->grad;
        out->prev[1]->grad -= out->grad;
    };

    out->_backward = sub_backward;
    return out;
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
    return a->operator-(b);
}

std::shared_ptr<Value> Value::operator*(std::shared_ptr<Value> other) {
    std::shared_ptr<Value> out = std::shared_ptr<Value>(new Value(this->data * other->data, {shared_from_this(), other}));

    std::function<void(void)> mul_backward = [out]() {
        out->prev[0]->grad += out->prev[1]->data * out->grad;
        out->prev[1]->grad += out->prev[0]->data * out->grad;
    };

    out->_backward = mul_backward;
    return out;
}

std::shared_ptr<Value> operator*(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
    return a->operator*(b);
}

std::shared_ptr<Value> Value::operator/(std::shared_ptr<Value> other) {
    std::shared_ptr<Value> out = std::shared_ptr<Value>(new Value(this->data / other->data, {shared_from_this(), other}));

    std::function<void(void)> div_backward = [out]() {
        out->prev[0]->grad += out->grad / out->prev[1]->data;
        out->prev[1]->grad -= out->prev[0]->data * out->grad / (out->prev[1]->data * out->prev[1]->data);
    };

    out->_backward = div_backward;
    return out;
}

std::shared_ptr<Value> operator/(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
    return a->operator/(b);
}

std::shared_ptr<Value> Value::pow(double exp) {
    std::shared_ptr<Value> out = std::shared_ptr<Value>(new Value(std::pow(this->data, exp), {shared_from_this()}));

    std::function<void(void)> pow_backward = [out, exp]() {
        out->prev[0]->grad += exp * std::pow(out->prev[0]->data, exp - 1) * out->grad;
    };

    out->_backward = pow_backward;
    return out;
}

std::shared_ptr<Value> Value::relu() {
    std::shared_ptr<Value> out = std::shared_ptr<Value>(new Value(std::max(0.0, this->data), {shared_from_this()}));

    std::function<void(void)> relu_backward = [out]() {
        out->prev[0]->grad += (out->prev[0]->data > 0) * out->grad;
    };

    out->_backward = relu_backward;
    return out;
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> a) {
    std::shared_ptr<Value> out = std::shared_ptr<Value>(new Value(-a->data, {a}));

    std::function<void(void)> neg_backward = [out]() {
        out->prev[0]->grad -= out->grad;
    };

    out->_backward = neg_backward;
    return out;
}

void Value::backward() {
    this->grad = 1.0;
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<std::shared_ptr<Value>> visited;

    std::function<void(std::shared_ptr<Value>)> build_topo = [&](std::shared_ptr<Value> v) {
        if (visited.count(v)) return;
        for (auto& child : v->prev) {
            build_topo(child);
        }
        topo.push_back(v);
        visited.insert(v);
    };

    build_topo(shared_from_this());
    std::reverse(topo.begin(), topo.end());

    for (auto& v : topo) {
        if (v->_backward) {
            v->_backward();
        }
    }
}