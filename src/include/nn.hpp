#pragma once
#include <vector>
#include <cstdlib>
#include "engine.hpp"

struct Module {
    Module() = default;
    void zero_grad() {
        for (auto& p: parameters()) {
            p->grad = 0.0;
        }
    }
    virtual std::vector<std::shared_ptr<Value>> parameters() = 0;
};

struct Neuron: public Module {
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;
    bool non_linear;

    Neuron() = default;
    Neuron(int in_features, bool non_linear);

    std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>>& x);

    std::vector<std::shared_ptr<Value>> parameters();
};

struct Layer: public Module {
    std::vector<Neuron> neurons;
    Layer() = default;
    Layer(int in_features, int out_features, bool non_linear);
    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>>& x);
    std::vector<std::shared_ptr<Value>> parameters();
};

struct MLP: public Module {
    std::vector<Layer> layers;
    MLP() = default;
    MLP(int in_features, std::vector<int> out_features);
    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>>& x);
    std::vector<std::shared_ptr<Value>> parameters();
};
