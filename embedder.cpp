#include "embedder.h"
#include <random>
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <fstream>

#include <iostream>

using std::vector;

double sig(double x) {
    return 1/(1+std::exp(-x));
}

double random(double min, double max) {
    return (double)std::rand()/(double)RAND_MAX*(max-min) + min;
}

Eigen::VectorXd softmax(Eigen::VectorXd input) {
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
        if (total != total) {
            printf("NAN in total");
        }
    }
    Eigen::VectorXd out(input.size());
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
    proj_mat = Eigen::MatrixXd::Random(embed_size, vocab);
    dec_mat = Eigen::MatrixXd::Random(vocab, embed_size);
}

Embedder::~Embedder() {

}

Eigen::VectorXd Embedder::predict(Eigen::VectorXd input) {
    //hidden_layer = input * proj_mat;
    hidden_layer = proj_mat * input;
    double max = abs(hidden_layer[0]);
    for (size_t i = 0; i < hidden_layer.size(); i++)
    {
        if (hidden_layer[i] != hidden_layer[i]) {
            printf("Nan in hidden layer");
        }
        if (max < abs(hidden_layer[i])) {
            max = abs(hidden_layer[i]);
        }
    }
    hidden_layer /= max;
    
    //output = hidden_layer * dec_mat;
    Eigen::VectorXd output = dec_mat * hidden_layer;
    return softmax(output);
}

double Embedder::train(Eigen::VectorXd input, Eigen::VectorXd expected, double rate) {
    Eigen::VectorXd output = predict(input);

    // error per output node
    // std::cout << total_error << std::endl;
    // Loss calculation
    double loss = 0;
    double sum_exp = 0;
    for (size_t i = 0; i < output.size(); i++)
    {
        if (expected[i] == 1) {
            loss -= std::log(output[i])*expected[i];
            if (loss != loss) {
                printf("Nan Loss");
            }
        }
    }


    // propagate errors


    Eigen::VectorXd da2 = output - expected;
    Eigen::MatrixXd dw2 = da2 * hidden_layer.transpose();
    Eigen::VectorXd da1 = (da2.transpose() * dec_mat);
    Eigen::MatrixXd dw1 = da1 * output.transpose();
    proj_mat -= rate * dw1;
    dec_mat -= rate * dw2;


    return loss;
}


void Embedder::serialize(const std::string& filename) {
    
}