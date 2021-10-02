#include <iostream>
#include "Tensor.h"
#include <chrono>
#include "assets.h"
#include "dataset.h"
#include "net.h"
#include <math.h>

#define OUTPUT_COUNT 3

int main(){   
    Tensor b({3,3}, 3);
    // Tensor c = Net::tanH(b);
    // Tensor x = Net::DerivativetanH(c);
    // c.printShape();
    // for(unsigned i=0; i<3; i++){
    //     for(unsigned j=0; j<3; j++){
    //         for(unsigned k=0; k<3; k++){
    //             std::cout << x({i,j,k}) << ", ";
    //         }
    //     }
    // }
    Tensor weights = Net::weightInit({3,4}, 3);
    Net::FeedForwardResult result = Net::feedForward(b, weights);
    result.net.printShape();
    result.output.printShape();

    // IrisDataset dataset("dataset/iris.data");
    // dataset.setType(IrisDataset::TRAIN, true);
    // IrisDataset::Sample sample = dataset.getSet();
    // sample.data.printShape();
    // sample.labels.printShape();
    // Tensor a = output_to_class(sample.labels);
    // for(size_t i=0; i<80; i++){
    //     std::cout << a({i}) << ", ";
    // }

    return 0;
}