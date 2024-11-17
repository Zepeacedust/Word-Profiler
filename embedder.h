#pragma once
#include <vector>
#include <string>
#include <eigen3/Eigen/Eigen>
#include "wordmapper.h"

using std::vector;

class Embedder {
    int vocab;
public:
    int embed_size;
    Eigen::MatrixXd proj_mat;
    Eigen::MatrixXd dec_mat;
    Eigen::VectorXd hidden_layer;
    Embedder(int _embed_size, int _vocab);
    Embedder(const std::string& filename);
    ~Embedder();
    Eigen::VectorXd predict(Eigen::VectorXd input);
    double train(Eigen::VectorXd input, Eigen::VectorXd expected, double rate);
    void batch();
    void serialize(const std::string& filename,  WordMapper & mapper);

};