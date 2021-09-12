#include <iostream>
#include "Tensor.h"
// #include "assets.h"
// #include "dataset.h"

#define OUTPUT_COUNT 3

int main(){
    Tensor b({2,2,2,2,3,4,6,23,7,3,4,2}, 0);
    Tensor a = b[{1,1,1,1,2,3,4}];
    a.printShape();
    // std::cout << a({1}) << std::endl;
    

    // Tensor cls = Tensor({3,2,2}, 3);
    // Tensor x = cls - 5.5;
    // std::cout << x({0,0,0});
    // std::cout << x({0,0,1});
    
    return 0;
}