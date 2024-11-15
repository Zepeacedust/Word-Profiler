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

vector<double> softmax(vector<double> input) {
    double total = 0;
    // filter for too large values

    double max = input[0];
    for (size_t i = 0; i < input.size(); i++)
    {
        if (input[i] > max)
        {
            max = input[i];
        }
        
    }
    

    for (size_t i = 0; i < input.size(); i++)
    {
        total += std::exp(input[i]-max);
    }
    vector<double> out(input.size());
    for (size_t i = 0; i < input.size(); i++)
    {
        out[i] = exp(input[i]-max)/total;
        if (out[i] != out[i]) {
            printf("Nan in softmax");
        }
    }
    return out;
}

Embedder::Embedder(int _embed_size, int _vocab) {
    embed_size = _embed_size;
    vocab = _vocab;
    for (size_t x = 0; x < embed_size; x++)
    {
        vector<double> line(vocab);
        for (size_t y = 0; y < vocab; y++)
        {
            line[y] = random(-1,1);
        }
        proj_mat.push_back(line);
    }

    for (size_t x = 0; x < vocab; x++)
    {
        vector<double> line(embed_size);
        for (size_t y = 0; y < embed_size; y++)
        {
            line[y] = random(-1,1);
        }
        dec_mat.push_back(line);
    }
}

Embedder::~Embedder() {

}

vector<double> Embedder::predict(vector<double> input) {
    //hidden_layer = input * proj_mat;
    hidden_layer = vector<double>(embed_size);
    for (size_t in_node = 0; in_node < vocab; in_node++)
    {
        if (input[in_node] == 0) {
            continue;
        }
        for (size_t node  = 0; node  < embed_size; node ++)
        {
            hidden_layer[node] += proj_mat[node][in_node];
            if (hidden_layer[node] != hidden_layer[node]) {
               printf("Nan in hidden layer");
            }
        }
    }
    
    //output = hidden_layer * dec_mat;
    vector<double> output(vocab);
    for (size_t hidden_node = 0; hidden_node < embed_size; hidden_node++)
    {
        for (size_t out_node = 0; out_node < vocab; out_node++)
        {
            output[out_node] += hidden_layer[hidden_node] * dec_mat[out_node][hidden_node];
            if (output[out_node] != output[out_node]) {
               printf("Nan in output");
            }
        }    
    }
    return softmax(output);
}

double Embedder::train(vector<double> input, vector<double> expected, double rate) {
    vector<double> output = predict(input);

    // error per output node
    double sum_1 = 0;
    vector<double> total_error(expected.size());
    int count_1s = 0;
    for (size_t i = 0; i < expected.size(); i++)
    {
        count_1s += 1;        
    }
    for (size_t i = 0; i < output.size(); i++)
    {
        if (expected[i] == 1){
            total_error[i]= (output[i]-1) + ( (count_1s -1) * output[i]);
        } else {
            total_error[i]= (count_1s * output[i]);
        }
    }

    // Loss calculation
    double loss = 0;
    double sum_exp = 0;
    for (size_t i = 0; i < output.size(); i++)
    {
        if (expected[i] == 1) {
            loss -= output[i];
        }
    }

    for (size_t i = 0; i < output.size(); i++)
    {
        sum_exp += std::exp(output[i]);
        if (sum_exp != sum_exp) {
            printf("Nan Exp");
        }
    }
    sum_exp = std::log(sum_exp);
    loss += sum_exp * sum_1;
    
    if (loss != loss) {
        printf("Nan Loss");
    }

    // propagate errors
    vector<double> hidden_errors(embed_size);
    for (size_t hidden_node = 0; hidden_node < embed_size; hidden_node++)
    {
        for (size_t output_node = 0; output_node < vocab; output_node++)
        {
            hidden_errors[hidden_node] += expected[output_node] * total_error[output_node] * dec_mat[output_node][hidden_node]; 
        }
    }
    
    //adjust weights
    for (size_t hidden_node = 0; hidden_node < embed_size; hidden_node++)
    {
        for (size_t output_node = 0; output_node < vocab; output_node++)
        {
            dec_mat[output_node][hidden_node] += -rate * dec_mat[output_node][hidden_node] * total_error[output_node];
        }
    }
    
    for (size_t hidden_node = 0; hidden_node < embed_size; hidden_node++)
    {
        for (size_t input_node = 0; input_node < vocab; input_node++)
        {
            proj_mat[hidden_node][input_node] += -rate * hidden_errors[hidden_node] * input[input_node]; 
        }
    }
    return loss;
}


void Embedder::serialize(const std::string& filename) {
    
}