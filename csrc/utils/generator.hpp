#ifndef GENERATOR_HPP
#define GENERATOR_HPP

#include <vector>
#include <utility>
#include <string>
#include <random>

#include "../Th/ThTypes.hpp"
#include "../Th/Th.hpp"

class Generator{
    public:
        int32_t _state;
        Generator(int32_t seed = std::random_device{}()) : engine(seed){}

        void manual_seed(int32_t seed){
            engine.seed(seed);
        }

        int32_t randint(int32_t start, int32_t end){
            std::uniform_int_distribution<int> dist(start, end);
            return dist(engine);
        }

        double_t randfloat(double_t start = 0.0, double_t end = 1.0){
            std::uniform_real_distribution<double_t> dist(start, end);
            return dist(engine);
        }

    private:
        std::mt19937 engine;

};

#endif