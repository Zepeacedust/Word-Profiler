#include <vector>
#include <string>
using std::vector;

class Embedder {
    vector<int> classifier_layers;
    int vocab;
    int embed_size;
public:
    vector<vector<double>> int_values;
    vector<vector<double>> mappings;
    vector<vector<vector<double>>> classifier;
    Embedder(int _embed_size, int _vocab, vector<int> classifier_layers);
    Embedder(const std::string& filename);
    ~Embedder();
    vector<double> predict(int a, int b);
    double train(int a, int b, vector<double> expected,double rate);
    void batch();
    void serialize(const std::string& filename);

};