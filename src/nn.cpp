#include "nn.hpp"

Neuron::Neuron(int in_features, bool non_linear) : non_linear{non_linear} {
    w.resize(in_features);
    for (int i = 0; i < in_features; i++) {
        w[i] = std::shared_ptr<Value>(new Value(static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0));
    }
    b = std::shared_ptr<Value>(new Value(0.0));
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
    std::vector<std::shared_ptr<Value>> params(w.size() + 1);
    for (int i = 0; i < w.size(); i++) {
        params[i] = w[i];
    }
    params[w.size()] = b;
    return params;
}

std::shared_ptr<Value> Neuron::operator()(std::vector<std::shared_ptr<Value>>& x) {
    std::shared_ptr<Value> out = std::make_shared<Value>(0.0);
    for (int i = 0; i < x.size(); i++) {
        out = out + w[i] * x[i];
    }
    out = out + b;
    if (non_linear) {
        out = out->relu();
    }
    return out;
}

Layer::Layer(int in_features, int out_features, bool non_linear) {
    neurons.resize(out_features);
    for (int i = 0; i < out_features; i++) {
        new(&neurons[i]) Neuron(in_features, non_linear);
    }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(std::vector<std::shared_ptr<Value>>& x) {
    std::vector<std::shared_ptr<Value>> out(neurons.size());
    for (int i = 0; i < neurons.size(); i++) {
        out[i] = neurons[i](x);
    }
    return out;
}

std::vector<std::shared_ptr<Value>> Layer::parameters() {
    std::vector<std::shared_ptr<Value>> params;
    for (int i = 0; i < neurons.size(); i++) {
        for (auto p: neurons[i].parameters()) {
            params.push_back(p);
        }
    }
    return params;
}

MLP::MLP(int in_features, std::vector<int> out_features) {
    int num_layers = out_features.size();

    layers.resize(num_layers);
    for (int i = 0; i < num_layers; i++) {
        if (i == 0) {
            new(&layers[i]) Layer(in_features, out_features[i], true);
        }
        else if (i == num_layers - 1) {
            new(&layers[i]) Layer(out_features[i - 1], out_features[i], false);
        }
        else {
            new(&layers[i]) Layer(out_features[i - 1], out_features[i], true);
        }
    }
}

std::vector<std::shared_ptr<Value>> MLP::parameters() {
    std::vector<std::shared_ptr<Value>> params;
    for (int i = 0; i < layers.size(); i++) {
        for (auto p: layers[i].parameters()) {
            params.push_back(p);
        }
    }
    return params;
}

std::vector<std::shared_ptr<Value>> MLP::operator()(std::vector<std::shared_ptr<Value>>& inputs) {
    std::vector<std::shared_ptr<Value>> x = inputs;
    for (int i = 0; i < layers.size(); i++) {
        x = layers[i](x);
    }
    return x;
}