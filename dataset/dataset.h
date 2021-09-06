#pragma once

#include <iostream>
#include <fstream>
#include "matrix.h"

class IrisDataset{
private:
    char* PATH;
    std::FILE *file;

    enum LINE_LENGTHS {
        FIRST_CLASS = 28, SECOND_CLASS = 32, THRIRD_CLASS = 31
    };

    Matrix getSample(unsigned int);

public:
    int INPUT_COUNT;
    int OUTPUT_COUNT;
    int TRAIN_COUNT;
    int TEST_COUNT;
    int VALIDATION_COUNT;

    IrisDataset(const char*);

    //Sets
    Matrix trainingSet(unsigned int);
    Matrix testSet(unsigned int);
    Matrix validationSet(unsigned int);
    void closeDataset();

    //Exception
    void indexOutOfRange();
};