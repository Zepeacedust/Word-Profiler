#include <iostream>
#include <fstream>
#include "tokenizer.h"
#include "wordmapper.h"

#include "embedder.h"

#include <vector>

#include <algorithm>

#include <random>

#define CONTEXT_LENGTH 5
#define NEG_SAMPLES 3

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

double dot(std::vector<double>& a,std::vector<double>& b ) {

    double prod = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        prod += a[i] * b[i];
    }
    return prod;
}
double cos_sim(std::vector<double>& a,std::vector<double>& b ) {

    double prod = dot(a,b);
    prod /= std::sqrt(dot(a,a)) * std::sqrt(dot(b,b));
    return prod;
}

// bool comp(std::vector<double> a, std::vector<double> b) {
//     return a[1] > b[1];
// }
// std::vector<std::vector<double>> find_closest(std::vector<double>& vec, Embedder& embedder) {
//     std::vector<std::vector<double>> pairs;

//     for (size_t i = 0; i < embedder.mappings.size(); i++)
//     {
//         pairs.push_back(std::vector<double>({(double)i ,cos_sim(vec, embedder.mappings[i])}));
//     }
//     std::sort(pairs.begin(), pairs.end(), comp);
//     return pairs; 
// }


// void demo_function(std::string a, std::string b, std::string c, Embedder& embedder, WordMapper& mapper) {
//     std::vector<double> a_vec = embedder.mappings[mapper.add_word(a)];
//     std::vector<double> b_vec = embedder.mappings[mapper.add_word(b)];
//     std::vector<double> c_vec = embedder.mappings[mapper.add_word(c)];
    
//     std::vector<double> con_vec = std::vector<double>(embedder.embed_size);

//     for (size_t i = 0; i < embedder.embed_size; i++)
//     {
//         con_vec[i] = a_vec[i] - b_vec[i] + c_vec[i];
//     }
    
//     std::vector<std::vector<double>> closest = find_closest(con_vec, embedder);

//     for (size_t i = 0; i < 10; i++)
//     {    
//         std::cout << mapper.get_word((int)closest[i][0]) << " " << closest[i][1] << std::endl;
//     }
    

// }


int main() {
    WordMapper word_mapper = WordMapper();
    Tokenizer tokenizer = Tokenizer("corpuses/sonnets.txt");
    std::vector<int> tokenized;
    std::vector<int> frequencies = std::vector<int>();
    std::vector<std::vector<int>> neighbors;
    
    std::cout << "Reading file" << std::endl;
    
    int context[CONTEXT_LENGTH];
    int context_ind = 0;
    std::fill(context, context+CONTEXT_LENGTH, -1);
    
    while (!tokenizer.empty) {

        std::string token = tokenizer.next_token();
        if (token == "") {
            continue;
        }
        int word_id = word_mapper.add_word(token);
        tokenized.push_back(word_id);
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

    Embedder embedder(50, word_mapper.size());

    
    int vocab = word_mapper.size();


    for (size_t epoch = 0; epoch < 10; epoch++)
    {
        std::cout << "Epoch " << epoch << std::endl;
        double total_loss = 0;
        for (size_t i = CONTEXT_LENGTH; i < tokenized.size()-CONTEXT_LENGTH; i++)
        {
            if (i%((tokenized.size()-CONTEXT_LENGTH * 2)/100) == 0) {
                std::cout << "#" << std::flush;
            }
            Eigen::VectorXd context_vec(vocab);
            Eigen::VectorXd expected_vec(vocab);
            for (size_t j = 1; j <= CONTEXT_LENGTH; j++)
            {
                context_vec[tokenized[i-j]] = 1;
                context_vec[tokenized[i+j]] = 1;
            }
            expected_vec[tokenized[i]] = 1;
            total_loss += embedder.train(context_vec, expected_vec, 0.1);
        }
        std::cout << std::endl;
        std::cout << "Average loss: " << total_loss / (tokenized.size()-2*CONTEXT_LENGTH) << std::endl;
    }
    
    while (true) {
        std::string word_1;
        std::cin >> word_1;
        int ind_1 = word_mapper.add_word(word_1);
        std::string word_2;
        std::cin >> word_2;
        int ind_2 = word_mapper.add_word(word_2);

        std::string word_3;
        std::cin >> word_3;
        int ind_3 = word_mapper.add_word(word_3);

        std::string word_4;
        std::cin >> word_4;
        int ind_4 = word_mapper.add_word(word_4);
        std::string word_5;
        std::cin >> word_5;
        int ind_5 = word_mapper.add_word(word_5);
        if (ind_1 >= vocab || ind_2 >= vocab || ind_3 >= vocab|| ind_4 >= vocab|| ind_5 >= vocab) {
            std::cout << "Word not recognized" << std::endl;    
            continue;
        }

        Eigen::VectorXd input(vocab);
        input[ind_1] = 1;
        input[ind_2] = 1;
        input[ind_4] = 1;
        input[ind_5] = 1;
        Eigen::VectorXd out = embedder.predict(input);

        for (size_t i = 0; i < out.size(); i++)
        {
            if (out[i] != 0) {
                std::cout << word_mapper.get_word(i) << "\t" << out[i] << std::endl;
            }
        }
        

        std::cout << out[ind_3] << std::endl;

    }   
    return 0;
}