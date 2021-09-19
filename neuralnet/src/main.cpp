#include <iostream>
#include "Tensor.h"
#include <chrono>
// #include "assets.h"
// #include "dataset.h"

#define OUTPUT_COUNT 3

int main(){   
    Tensor b({3,2,2,2}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}, {}, true);
    Tensor c({3,2,2,2}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}, {}, true);
    Tensor x = b.concat(c, 3);
    x.printShape();
    x.printStrides();

    for(unsigned i=0; i<x.totalLength; i++){
        std::cout << x.getValues()[i] << ", ";
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