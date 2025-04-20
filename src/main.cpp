#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include "engine.hpp"
#include "nn.hpp"

using namespace std;


int main() {
    ifstream is("data.txt");
    if (!is) {
        cerr << "Failed to open file" << endl;
        return 1;
    }
    vector<pair<vector<shared_ptr<Value>>, shared_ptr<Value>>> data;
    double x1, x2, y;
    while (is >> x1 >> x2 >> y) {
        data.push_back(
            make_pair<vector<shared_ptr<Value>>, shared_ptr<Value>>({shared_ptr<Value>(new Value(x1)),
            shared_ptr<Value>(new Value(x2))}, shared_ptr<Value>(new Value(y))));
    }

    MLP model(2, {16, 16, 1}); // 2-layers neural network
    std::cout << "num of parameters: " << model.parameters().size() << std::endl;

    // training loop
    for (int i = 0; i < 100; i++) {
        shared_ptr<Value> loss = make_shared<Value>(0.0);
        double accuracy = 0.0;
        for (auto& [x, y] : data) {
            auto pred = model(x);
            // loss = max(0, 1 - y * pred)
            loss = loss + (shared_ptr<Value>(new Value(1.0)) - y * pred[0])->relu();

            if (pred[0]->data * y->data > 0) {
                accuracy += 1.0;
            }
        }

        accuracy /= static_cast<double>(data.size());
        accuracy *= 100.0;
        loss = loss / shared_ptr<Value>(new Value(static_cast<double>(data.size())));
        loss->backward();

        double learning_data = 1.0 - 0.9 * i / 100; 
        for (auto& p : model.parameters()) {
            p->data -= 0.01 * p->grad;
        }

        std::cout << "step " << i << " loss: " << loss->data << ", accuracy: " << accuracy << "%" << std::endl;
    }
    return 0;
}