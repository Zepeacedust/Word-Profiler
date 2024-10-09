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

Embedder::Embedder(int _embed_size, int _vocab, vector<int> classifier_layers) {

    embed_size = _embed_size;
    vocab = _vocab;
    this->classifier_layers = classifier_layers;

    for (size_t i = 0; i < vocab; i++)
    {
        vector<double> embedding;
        for (size_t x = 0; x < embed_size; x++)
        {
            embedding.push_back(random(-1,1));
        }
        mappings.push_back(embedding);
    }

    for (size_t layer_nr = 0; layer_nr < classifier_layers.size(); layer_nr++)
    {
        vector<vector<double>> layer;
        for (size_t node_id = 0; node_id < classifier_layers[layer_nr]; node_id++)
        {
            vector<double> node;
            int lower_layer;
            if (layer_nr == 0) {
                lower_layer = 2 * embed_size;
            } else {
                lower_layer = classifier_layers[layer_nr-1];
            }
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

vector<double> Embedder::predict(int a, int b) {
    for (size_t node_id = 0; node_id < classifier_layers[0]; node_id++)
    {
        double acc = 0;
        for (size_t downstream = 0; downstream < embed_size; downstream++)
        {
            acc += sig(mappings[a][downstream]) * classifier[0][node_id][downstream];
            acc += sig(mappings[b][downstream]) * classifier[0][node_id][downstream+embed_size];
        }
        int_values[0][node_id] = sig(acc);
    }
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

double Embedder::train(int a, int b, vector<double> expected, double rate) {
    
    vector<double> output = predict(a, b);

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
                    * classifier[layer][upstream][node_id] 
                    * errors[upstream];
                classifier[layer+1][upstream][node_id] += 
                      -(int_values[layer][node_id])
                    * errors[upstream] * rate;
            }
            new_errors.push_back(acc);
        }
        errors = new_errors;
    }

    

    for (size_t upstream = 0; upstream < embed_size; upstream++)
    {
        double a_error = 0;
        double b_error = 0;
        for (size_t node_id = 0; node_id < classifier_layers[0]; node_id++)
        {
            a_error += 
                  (sig(mappings[a][upstream]))
                * (1-sig(mappings[a][upstream])) 
                * classifier[0][node_id][upstream] 
                * errors[node_id];

            b_error += 
                  (sig(mappings[b][upstream]))
                * (1-sig(mappings[b][upstream])) 
                * classifier[0][node_id][upstream+embed_size] 
                * errors[node_id];
            if (b_error != b_error || a_error != a_error) {
                b_error = 2*b_error;
            }
            classifier[0][node_id][upstream] += -rate * mappings[a][upstream] * errors[node_id];
            classifier[0][node_id][upstream+embed_size] += -rate * mappings[b][upstream] * errors[node_id];
        }
        mappings[a][upstream] -= a_error;
        mappings[b][upstream] -= b_error;
    }
    
    


    // TODO: adjust vectors
    return error;
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

    size = mappings.size();

    outfile.write((const char *) &embed_size, sizeof(double));
    outfile.write((const char *) &size, sizeof(double));

    for (size_t mapping = 0; mapping < size; mapping++)
    {
        for (size_t ind = 0; ind < embed_size; ind++)
        {
            outfile.write((const char *) &mappings[mapping][ind], sizeof(double));
        }
    }
}