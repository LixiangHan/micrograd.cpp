#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <memory>


struct Value: public std::enable_shared_from_this<Value> {
    double data;
    double grad;
    std::vector<std::shared_ptr<Value>> prev;
    std::function<void(void)> _backward;

    Value() : data{0.0}, grad{0.0}, prev{} {};
    Value(double);
    Value(double, std::vector<std::shared_ptr<Value>>);

    std::shared_ptr<Value> operator+(std::shared_ptr<Value> other);
    
    std::shared_ptr<Value> operator-(std::shared_ptr<Value> other);
    
    std::shared_ptr<Value> operator*(std::shared_ptr<Value> other);
    
    std::shared_ptr<Value> operator/(std::shared_ptr<Value> other);
    
    std::shared_ptr<Value> pow(double);

    std::shared_ptr<Value> relu();

    void backward();
};

std::shared_ptr<Value> operator+(std::shared_ptr<Value> a, std::shared_ptr<Value> b);

std::shared_ptr<Value> operator-(std::shared_ptr<Value> a, std::shared_ptr<Value> b);

std::shared_ptr<Value> operator*(std::shared_ptr<Value> a, std::shared_ptr<Value> b);

std::shared_ptr<Value> operator/(std::shared_ptr<Value> a, std::shared_ptr<Value> b);

std::shared_ptr<Value> operator-(std::shared_ptr<Value> a);