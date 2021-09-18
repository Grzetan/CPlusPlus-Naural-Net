#include <iostream>
#include "Tensor.h"
#include <chrono>
// #include "assets.h"
// #include "dataset.h"

#define OUTPUT_COUNT 3

int main(){   
    Tensor b({2,2,2,3}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}, {}, true);
    b = b.transpose();
    b = b.flatten();
    b.printStrides();
    b.printShape();
    for(size_t i=0; i<24; i++){
        std::cout << b({i}) << ", ";
    }
    std::cout << std::endl;


    // b.printShape();
    // std::cout << a({1}) << std::endl;
    

    // Tensor cls = Tensor({3,2,2}, 3);
    // Tensor x = cls - 5.5;
    // std::cout << x({0,0,0});
    // std::cout << x({0,0,1});
    
    return 0;
}