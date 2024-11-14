#include <iostream>
#include <fstream>
#include "tokenizer.h"
#include "wordmapper.h"

#include "embedder.h"

#include <vector>

#include <algorithm>

#include <random>

#define CONTEXT_LENGTH 4
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

bool comp(std::vector<double> a, std::vector<double> b) {
    return a[1] > b[1];
}
std::vector<std::vector<double>> find_closest(std::vector<double>& vec, Embedder& embedder) {
    std::vector<std::vector<double>> pairs;

    for (size_t i = 0; i < embedder.mappings.size(); i++)
    {
        pairs.push_back(std::vector<double>({(double)i ,cos_sim(vec, embedder.mappings[i])}));
    }
    std::sort(pairs.begin(), pairs.end(), comp);
    return pairs; 
}


void demo_function(std::string a, std::string b, std::string c, Embedder& embedder, WordMapper& mapper) {
    std::vector<double> a_vec = embedder.mappings[mapper.add_word(a)];
    std::vector<double> b_vec = embedder.mappings[mapper.add_word(b)];
    std::vector<double> c_vec = embedder.mappings[mapper.add_word(c)];
    
    std::vector<double> con_vec = std::vector<double>(embedder.embed_size);

    for (size_t i = 0; i < embedder.embed_size; i++)
    {
        con_vec[i] = a_vec[i] - b_vec[i] + c_vec[i];
    }
    
    std::vector<std::vector<double>> closest = find_closest(con_vec, embedder);

    for (size_t i = 0; i < 10; i++)
    {    
        std::cout << mapper.get_word((int)closest[i][0]) << " " << closest[i][1] << std::endl;
    }
    

}


int main() {
    WordMapper word_mapper = WordMapper();
    Tokenizer tokenizer = Tokenizer("corpuses/shakes.txt");
    std::vector<int> tokenized;
    std::vector<int> frequencies = std::vector<int>();
    std::vector<std::vector<int>> neighbors;
    
    int context[CONTEXT_LENGTH];
    int context_ind = 0;
    std::fill(context, context+CONTEXT_LENGTH, -1);

    while (!tokenizer.empty) {

        std::string token = tokenizer.next_token();
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

    Embedder test(50, word_mapper.size(), vector<int>({1}));

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
            loss += test.train(training_data[ind][0], training_data[ind][1], {1}, .01);
            for (size_t i = 0; i < NEG_SAMPLES; i++)
            {
                int neg_sample = tokenized[rand()%tokenized.size()];
                bool present = false;
                for (size_t i = 0; i < neighbors[training_data[ind][0]].size(); i++)
                {
                    if (neg_sample == neighbors[training_data[ind][0]][i]) {
                        present = true;
                    }
                }
                if(!present) {
                    loss += test.train(training_data[ind][0], tokenized[rand()%tokenized.size()], {0}, .01);
                }
            }
            
        }
        std ::cout << std::endl;
        std::cout << loss << "  average: " << loss /((double)training_data.size()*(1+NEG_SAMPLES)) << std::endl;
        std::cout << "Bias: " << test.biases[0][0];
    }


    test.serialize("embedder.dat");


    while (true) {
        std::string word_1;
        std::cin >> word_1;
        std::string word_2;
        std::cin >> word_2;
        std::string word_3;
        std::cin >> word_3;
        int ind_1 = word_mapper.add_word(word_1);
        int ind_2 = word_mapper.add_word(word_2);
        int ind_3 = word_mapper.add_word(word_3);
        if (ind_1 >= vocab || ind_2 >= vocab || ind_3 >= vocab) {
            std::cout << "Word not recognized" << std::endl;    
            continue;
        }

        demo_function(word_1, word_2, word_3, test, word_mapper);
        
        
        std::cout << "Similarity: " << cos_sim(test.mappings[ind_1], test.mappings[ind_2]) << std::endl;
        std::cout << "Neighborlyness: " << test.predict(ind_1, ind_2)[0] << std::endl;
    }

    return 0;
}