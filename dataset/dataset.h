#pragma once

#include <iostream>
#include <fstream>
#include "Tensor.h"

class IrisDataset{
private:
    char* PATH;
    std::FILE *file;

    enum LINE_LENGTHS {
        FIRST_CLASS = 28, SECOND_CLASS = 32, THRIRD_CLASS = 31
    };

    Tensor getSample(unsigned int);

public:
    size_t INPUT_COUNT;
    size_t OUTPUT_COUNT;
    size_t TRAIN_COUNT;
    size_t TEST_COUNT;
    size_t VALIDATION_COUNT;

    IrisDataset(const char*);

    //Sets
    Tensor trainingSet(unsigned int);
    Tensor testSet(unsigned int);
    Tensor validationSet(unsigned int);
    void closeDataset();

    int classNameToIndex(char *);

    //Exception
    void indexOutOfRange();
    void classNotFound();
};