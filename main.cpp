#include <iostream>
#include <fstream>
#include "tokenizer.h"
#include "wordmapper.h"

#include "embedder.h"

#include <vector>

#include <algorithm>

#include <random>

static const int CONTEXT_LENGTH = 2;

bool includes(int a, const std::vector<int>& list) {
    for (size_t i = 0; i < list.size(); i++)
    {
        if (a == list[i]) return true; 
    }
    return false;
}

void describe(int word, WordMapper& word_mapper, std::vector<std::vector<int>>& neighbors) {
    std::cout << "Current word: \"" << word_mapper.get_word(word) << "\" token: " << word<< std::endl;
    for (size_t i = 0; i < neighbors[word].size(); i++) {
        std::cout << word_mapper.get_word(neighbors[word][i]) << std::endl;
    }
}
template<typename t>
void shuffle(vector<t>& shuffled) {
    for (size_t i = 0; i < shuffled.size()-1; i++)
    {
        int ind = random()%(shuffled.size()-i) + i; 
        t temp = shuffled[ind];
        shuffled[ind] = shuffled[i];
        shuffled[i] = temp;
    }
}

int main() {
    WordMapper word_mapper = WordMapper();
    Tokenizer tokenizer = Tokenizer("The Fellowship Of The Ring.txt");
    std::vector<int> frequencies = std::vector<int>();
    std::vector<std::vector<int>> neighbors;
    
    std::vector<int> tokenized;

    int context[CONTEXT_LENGTH];
    
    while (!tokenizer.empty) {
        std::string token = tokenizer.next_token();
        int word_id = word_mapper.add_word(token);
        if (frequencies.size()<= word_id) {
            frequencies.push_back(1);
        } else {
            frequencies[word_id] += 1;
        }
        tokenized.push_back(word_id);
    }

    std::vector<std::vector<int>> bags;
    for (size_t i = CONTEXT_LENGTH; i < tokenized.size()-CONTEXT_LENGTH; i++)
    {
        vector<int> bag;
        for (int j = -CONTEXT_LENGTH; j <= CONTEXT_LENGTH; j++)
        {
            bag.push_back(tokenized[i+j]);
        }
        bags.push_back(bag);
    }
    

    // for (size_t i = 0; i < frequencies.size(); i++){
    //     std::cout << frequencies[i] << " \t " << word_mapper.get_word(i) << std::endl;
    // }

    int vocab = word_mapper.size();
    
    Embedder test(word_mapper.size(), vector<int>({50, vocab}));

    for (size_t epoch = 0; epoch < 25; epoch++)
    {
        double loss = 0;
        std::cout << "epoch " << epoch << std::endl;
        shuffle(bags);

        for (size_t ind= 0; ind < bags.size(); ind++)
        {

            if (ind%(bags.size()/100) == 0) {
                std::cout << "#" << std::flush;
            }
            // std::cout << neighbors[a].size() << std::endl;

            vector<double> expected(vocab, 0);
            vector<double> input(vocab, 0);

            for (size_t i = 0; i < CONTEXT_LENGTH*2+1; i++)
            {
                if (i == CONTEXT_LENGTH){ 
                    expected[bags[ind][i]] = 1;
                }
                else {
                    input[bags[ind][i]] = 1;
                } 

            }
            

            loss += test.train(input, expected, .1);
        }
        std ::cout << std::endl;
        std::cout << loss << "  average: " << loss /((double)bags.size()) << std::endl;
    }


    test.serialize("embedder.dat");
    
    while (true) {
        std::string word_1;
        std::cin >> word_1;
        std::string word_2;
        std::cin >> word_2;
        int ind_1 = word_mapper.add_word(word_1);
        int ind_2 = word_mapper.add_word(word_2);
        if (ind_1 >= vocab || ind_2 >= vocab) {
            std::cout << "Word not recognized" << std::endl;    
            continue;
        }

    }

    return 0;
}