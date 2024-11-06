#include <vector>
#include <string>
using std::vector;

class Embedder {
    vector<int> network_layout;
    int vocab;
public:
    vector<vector<double>> int_values;
    vector<vector<vector<double>>> classifier;
    Embedder(int _vocab, vector<int> classifier_layers);
    Embedder(const std::string& filename);
    ~Embedder();
    vector<double> predict(const vector<double> &input);
    double train(const vector<double> &input, const vector<double> &expected, double rate);
    void batch();
    void serialize(const std::string& filename);

};