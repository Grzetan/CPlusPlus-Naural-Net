#include <iostream>
#include "Tensor.h"
#include <chrono>
// #include "assets.h"
// #include "dataset.h"

#define OUTPUT_COUNT 3

int main(){   
    Tensor b({2,2,2,3}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}, {});
    b.printStrides();
    b = b.permute({1,3,2,0});
    b.printStrides();
    b.printShape();
    for(size_t i=0; i<2; i++){
        for(size_t j=0; j<3; j++){
            for(size_t k=0; k<2; k++){
                for(size_t l=0; l<2; l++){
                    std::cout << b({i,j,k,l}) << std::endl;
                }
            }
        }
    }

    // b.printShape();
    // std::cout << a({1}) << std::endl;
    

    // Tensor cls = Tensor({3,2,2}, 3);
    // Tensor x = cls - 5.5;
    // std::cout << x({0,0,0});
    // std::cout << x({0,0,1});
    
    return 0;
}