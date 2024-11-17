#pragma once
#include <vector>

#include "binarytrie.h"


class WordMapper {
private:
    int _size = 0;
    std::vector<std::string> int_to_word;
    BinaryTrie word_to_int;
public:
    int size();
    WordMapper();
    ~WordMapper();
    int add_word(const std::string& s);
    std::string get_word(int i);
};