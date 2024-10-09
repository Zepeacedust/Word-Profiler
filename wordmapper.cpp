#include "wordmapper.h"

int WordMapper::size() {
    return _size;
}
WordMapper::WordMapper(){
}
WordMapper::~WordMapper(){

}

int WordMapper::add_word(const std::string& s){
    int id = word_to_int.add(s, 0, _size);
    if (id == _size) {
        _size += 1;
        int_to_word.push_back(s);
    } 
    return id;
    
}

std::string WordMapper::get_word(int i){
    return int_to_word[i];
}