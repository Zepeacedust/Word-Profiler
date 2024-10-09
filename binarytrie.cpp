#include "binarytrie.h"

int BinaryTrie::get(const std::string& s, int at) {
    if (s[at]==ch) {
        if (s.length() == at + 1) return id;
        if (down == nullptr) return -1;
        return down->get(s, at+1);
    }
    if (s[at] < ch) {
        if (left == nullptr) return -1;
        return left->get(s, at);
    }
    if (s[at] > ch) {
        if (right == nullptr) return -1;
        return right->get(s, at);
    }
    return -1;
};

int  BinaryTrie::add(const std::string& s, int at, int _id){
    if (ch == 0) {
        ch = s[at];
    }


    if (s[at] < ch) {
        if (left == nullptr) {
            left = new BinaryTrie();
            left->add(s, at, _id);
        }
        return left->add(s, at, _id);
    }

    if (s[at] > ch) {
        if (right == nullptr) {
            right = new BinaryTrie();
            right->add(s, at, _id);
        }
        return right->add(s, at, _id);
    }

    if (ch == s[at]) {
        if (at == s.length()-1) {
            if (id == -1) {
                id =_id;
            }
            return id;
        } else {
            if (down == nullptr) {
                down = new BinaryTrie();
            }
            return down->add(s, at+1, _id);
        }
    }

    return -1;
};

BinaryTrie::BinaryTrie(){

};

BinaryTrie::~BinaryTrie(){
    delete left;
    delete right;
    delete down;
};