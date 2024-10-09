#include <iostream>
#include <fstream>
#include "tokenizer.h"
#include "wordmapper.h"

#include "embedder.h"

#include <vector>

#include <algorithm>

#include <random>

#define CONTEXT_LENGTH 2

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
    Tokenizer tokenizer = Tokenizer("shakes.txt");
    std::vector<int> frequencies = std::vector<int>();
    std::vector<std::vector<int>> neighbors;
    
    int context[CONTEXT_LENGTH];
    int context_ind = 0;
    std::fill(context, context+CONTEXT_LENGTH, -1);

    while (!tokenizer.empty) {

        std::string token = tokenizer.next_token();
        int word_id = word_mapper.add_word(token);

        if (frequencies.size()<= word_id) {
            frequencies.push_back(1);
        } else {
            frequencies[word_id] += 1;
        }



        if (neighbors.size() <= word_id) {
            neighbors.push_back(std::vector<int>());
        }
        
        for (size_t i = 0; i < CONTEXT_LENGTH; i++)
        {
            if (context[i] == -1) {break;};
            neighbors[context[i]].push_back(word_id);
            neighbors[word_id].push_back(context[i]);
        }
        context[context_ind] = word_id;
        context_ind = (context_ind + 1) % CONTEXT_LENGTH; 
    }

    // for (size_t i = 0; i < frequencies.size(); i++){
    //     std::cout << frequencies[i] << " \t " << word_mapper.get_word(i) << std::endl;
    // }

    Embedder test(150, word_mapper.size(), vector<int>({1}));

    vector<vector<int>> training_data;
    
    for (int a = 0; a < word_mapper.size(); a++)
    {
        for (int b = 0; b < neighbors[a].size(); b++)
        {
            training_data.push_back({a,neighbors[a][b]});
        }
        
    }
    

    int vocab = word_mapper.size();
    

    for (size_t i = 0; i < 25; i++)
    {
        double loss = 0;
        std::cout << "epoch " << i << std::endl;
        shuffle(training_data);

        for (size_t ind= 0; ind < training_data.size(); ind++)
        {

            if (ind%(training_data.size()/100) == 0) {
                std::cout << "#" << std::flush;
            }
            // std::cout << neighbors[a].size() << std::endl;
            loss += test.train(training_data[ind][0], training_data[ind][1], {1}, .1);
            loss += test.train(training_data[ind][0], rand()%word_mapper.size(), {0}, .1);
            loss += test.train(training_data[ind][0], rand()%word_mapper.size(), {0}, .1);
        }
        std ::cout << std::endl;
        std::cout << loss << "  average: " << loss /((double)training_data.size()*3) << std::endl;
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

        std::cout << test.predict(ind_1, ind_2)[0] << std::endl;
    }

    return 0;
}