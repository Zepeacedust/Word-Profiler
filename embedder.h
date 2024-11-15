#include <vector>
#include <string>
using std::vector;

class Embedder {
    int vocab;
public:
    int embed_size;
    vector<vector<double>> proj_mat;
    vector<vector<double>> dec_mat;
    vector<double> hidden_layer;
    Embedder(int _embed_size, int _vocab);
    Embedder(const std::string& filename);
    ~Embedder();
    vector<double> predict(vector<double> input);
    double train(vector<double> input, vector<double> expected,double rate);
    void batch();
    void serialize(const std::string& filename);

};