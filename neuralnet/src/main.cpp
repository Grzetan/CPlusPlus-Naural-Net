#include <iostream>
#include <vector>
#include "matrix.h"
#include "assets.h"
#include "dataset.h"

#define OUTPUT_COUNT 3

int main(){
    IrisDataset dataset("dataset/iris.data");
    dataset.trainingSet(0);
}