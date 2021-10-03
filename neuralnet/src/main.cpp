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
    IrisDataset::Sample set = dataset.getSet();
    Tensor weights = Net::weightInit({5,3}, 0.5);
    Tensor bias({set.data.getShape()[0],1}, 1);
    Tensor targetCls = output_to_class(set.labels);
    Net::EvaluateErrors errors = Net::evaluate(set.data, weights, set.labels, targetCls, bias);
    std::cout << errors.outputError << ", " << errors.classificationError << std::endl;

    return 0;
}