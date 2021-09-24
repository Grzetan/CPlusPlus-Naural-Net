#include <iostream>
#include "Tensor.h"
#include <chrono>
// #include "assets.h"
// #include "dataset.h"

#define OUTPUT_COUNT 3

int main(){   
    Tensor b({2,4}, {0,1,2,3,4,5,6,7}, {}, true);
    Tensor c({4,2}, {0,1,2,3,4,5,6,7}, {}, true);
    Tensor x = b.kron(c);
    // x.printShape();
    // x.printStrides();
    // x.printStrides();
    x.printShape();
    x.printStrides();
    for(unsigned i=0; i<8; i++){
        for(unsigned j=0; j<8; j++){
            // for(unsigned k=0; k<4; k++){
                std::cout << x({i,j}) << ", ";
            // }
        }
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