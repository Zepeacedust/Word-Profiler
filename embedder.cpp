#include "embedder.h"
#include <random>
#include <cmath>

#include <fstream>


using std::vector;

double sig(double x) {
    return 1/(1+std::exp(-x));
}

double random(double min, double max) {
    return (double)std::rand()/(double)RAND_MAX*(max-min) + min;
}

Embedder::Embedder(int _vocab, vector<int> classifier_layers) {
    vocab = _vocab;
    this->classifier_layers = classifier_layers;
    
    for (size_t layer_nr = 0; layer_nr < classifier_layers.size(); layer_nr++)
    {
        vector<vector<double>> layer;
        for (size_t node_id = 0; node_id < classifier_layers[layer_nr]; node_id++)
        {
            vector<double> node;
            int lower_layer = classifier_layers[layer_nr-1];
            if (layer_nr == 0) lower_layer = vocab; 
            for (size_t downstream = 0; downstream < lower_layer; downstream++)
            {
                node.push_back(random(-1,1));
            }
            layer.push_back(node);
        }
        classifier.push_back(layer);
    }


    for (size_t layer = 0; layer < classifier_layers.size(); layer++)
    {
        int_values.push_back(vector<double>());
        for (size_t node_id = 0; node_id < classifier_layers[layer]; node_id++)
        {
            int_values[layer].push_back(0);
        }
    }
}

Embedder::~Embedder() {

}

vector<double> Embedder::predict(const vector<double> &input) {
    int_values[0] = input;
    // hidden layers
    for (size_t layer = 1; layer < classifier_layers.size(); layer++)
    {
        for (size_t node_id = 0; node_id < classifier_layers[layer]; node_id++)
        {
            double acc = 0;
            for (size_t downstream = 0; downstream < classifier_layers[layer-1]; downstream++)
            {
                acc += int_values[layer-1][downstream] * classifier[layer][node_id][downstream];
            }
            int_values[layer][node_id] = sig(acc);
        }
    }
    return int_values[classifier_layers.size()-1];
}

double Embedder::train(const vector<double> &input, const vector<double> &expected, double rate) {
    
    vector<double> output = predict(input);

    double error = 0;
    for (size_t i = 0; i < expected.size(); i++)
    {
        error += (output[i] - expected[i]) * (output[i] - expected[i]);
    }
    

    vector<double> errors;

    // slightly different formula for first layer
    for (size_t node_id = 0; node_id < classifier_layers[classifier_layers.size()-1]; node_id++)
    {
        errors.push_back((output[node_id] - expected[node_id]) * output[node_id] * (1 - output[node_id]));
    }    
    

    for (int layer = classifier_layers.size() - 2; layer >= 0; layer--)
    {
        vector<double> new_errors;
        for (size_t node_id = 0; node_id < classifier_layers[layer]; node_id++)
        {
            double acc = 0;
            for (size_t upstream = 0; upstream < classifier_layers[layer+1]; upstream++)
            {
                acc += 
                      (int_values[layer][node_id])
                    * (1-int_values[layer][node_id]) 
                    * classifier[layer+1][upstream][node_id] 
                    * errors[upstream];
                classifier[layer+1][upstream][node_id] += 
                      -(int_values[layer][node_id])
                    * errors[upstream] * rate;
            }
            new_errors.push_back(acc);
        }
        errors = new_errors;
    }
    // TODO: adjust vectors
    return error;
}

vector<double> softmax(const vector<double> &l) {
    double total = 0;
    for (size_t i = 0; i < l.size(); i++)
    {
        total += std::exp(l[i]);
    }
    vector<double> scaled(l);
    for (size_t i = 0; i < l.size(); i++)
    {
        scaled[i] = std::exp(scaled[i]) / total;
    }
    return scaled;
}

void Embedder::serialize(const std::string& filename) {
    std::ofstream outfile(filename.c_str(), std::ios::binary);

    std::size_t size = classifier_layers.size();

    outfile.write((const char *) &size, sizeof(std::size_t));

    for (size_t i = 0; i < size; i++)
    {
        outfile.write((const char *) &classifier_layers[i], sizeof(double));
    }

    for (size_t layer = 0; layer < size; layer++)
    {
        for (size_t node_id = 0; node_id < classifier[layer].size(); node_id++)
        {    
            for (size_t downstream = 0; downstream < classifier[layer][node_id].size(); downstream++)
            {
                outfile.write((const char *) &classifier[layer][downstream], sizeof(double));
            }
        }
    }
}