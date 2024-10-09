#include <string>

#include <fstream>

class Tokenizer
    {
private:
    char ch;
    std::ifstream file;
    char next_char();
public:
    bool empty = false;
    Tokenizer(const std::string& filename);
    std::string next_token();
};