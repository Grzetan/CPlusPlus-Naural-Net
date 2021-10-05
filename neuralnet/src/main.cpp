#include <iostream>
#include "Tensor.h"
#include <chrono>
#include "assets.h"
#include "dataset.h"
#include "net.h"
#include <math.h>

#define OUTPUT_COUNT 3

int main(){   
    IrisDataset dataset("dataset/iris.data");
    dataset.setType(IrisDataset::TRAIN, true);
    IrisDataset::Sample trainSet = dataset.getSet();

    dataset.setType(IrisDataset::TEST, true);
    IrisDataset::Sample testSet = dataset.getSet();

    dataset.setType(IrisDataset::VALIDATION, true);
    IrisDataset::Sample validationSet = dataset.getSet();    

    Net::train(trainSet, testSet, validationSet);
    return 0;
}