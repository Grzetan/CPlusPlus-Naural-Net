#include <iostream>
#include "Tensor.h"
#include <chrono>
#include "assets.h"
#include "dataset.h"

#define OUTPUT_COUNT 3

int main(){   
    // Tensor b({3}, 3);
    // Tensor a({3}, 4);
    // b.insertTensor({0}, a);

    // for(unsigned i=0; i<3; i++){
    //     // for(unsigned j=0; j<3; j++){
    //         std::cout << b({i}) << ", ";
    //     // }
    //     std::cout << std::endl;
    // }
    IrisDataset dataset("dataset/iris.data");
    dataset.setType(IrisDataset::TRAIN);
    IrisDataset::Sample sample = dataset.getSet();
    sample.data.printShape();
    sample.labels.printShape();
    for(size_t i=0; i<80; i++){
        for(size_t j=0; j<4; j++){
            std::cout << sample.data({i,j}) << ", ";
        }
        std::cout << std::endl;
    }

    for(size_t i=0; i<80; i++){
        for(size_t j=0; j<3; j++){
            std::cout << sample.labels({i,j}) << ", ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}