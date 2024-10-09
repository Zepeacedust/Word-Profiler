#include <string>

class BinaryTrie
{
private:
    BinaryTrie* down = nullptr;
    BinaryTrie* left = nullptr;
    BinaryTrie* right = nullptr;
    char ch = 0;
    int id = -1;
public:
    int get(const std::string& s, int at);
    int add(const std::string& s, int at, int _id);
    BinaryTrie();
    ~BinaryTrie();
};

