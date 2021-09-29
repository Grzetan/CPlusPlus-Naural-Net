#pragma once

#include <iostream>
#include <fstream>
#include "Tensor.h"

class IrisDataset{
public:
    enum Sets{
        TRAIN, TEST, VALIDATION
    };

    struct Sample{
        Tensor data;
        Tensor labels;
    };
private:
    char* PATH;
    size_t INPUT_COUNT = 4;
    size_t OUTPUT_COUNT = 3;
    std::FILE *file;
    unsigned indexes[150];
    Sets usedType;
    vector<unsigned> samples;
    unsigned batchSize;

    //Exception
    void indexOutOfRange();
    void classNotFound();

public:
    size_t TRAIN_COUNT = 80;
    size_t TEST_COUNT = 35;
    size_t VALIDATION_COUNT = 35;
    size_t TOTAL_COUNT = 150;

    IrisDataset(const char*);

    //Sets
    Sample getSample(unsigned int);
    Sample getSet();
    void setType(Sets);
    void closeDataset();

    int classNameToIndex(char *);
};