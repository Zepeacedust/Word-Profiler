#include "tokenizer.h"
#include <cctype>

bool is_ignored(char c) {
    switch (c) {
    case ' ':
    case '\n':
    case '\t':
    case ',':
    case '.':
    case ';':
    case ':':
    case '?':
        return true;
    default: 
        return false;
    }
}

Tokenizer::Tokenizer(const std::string& filename) {
    file = std::ifstream(filename);
    if (file.bad()) {
        std::printf("File bad");
    }
};

std::string Tokenizer::next_token() {
    std::string word;
    while (is_ignored(ch) && !empty) {
        next_char();
    }
    
    while (!is_ignored(ch) && !empty) {
        word += next_char();
    }

    return word;
}

char Tokenizer::next_char() {
    if (ch == 0) {
        file.read(&ch, 1);
    }
    char out = ch;
    file.read(&ch, 1);
    if (file.eof()) {
        empty = true;
    }

    return std::tolower(out);
}