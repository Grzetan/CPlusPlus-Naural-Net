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
    std::FILE *file;
    unsigned indexes[150];
    Sets usedType;
    bool bias;
    vector<unsigned> samples;
    unsigned batchSize;

    //Exception
    void indexOutOfRange();
    void classNotFound();
     
    static const size_t INPUT_COUNT = 4;
    static const size_t OUTPUT_COUNT = 3;
    static const size_t TRAIN_COUNT = 80;
    static const size_t TEST_COUNT = 35;
    static const size_t VALIDATION_COUNT = 35;
    static const size_t TOTAL_COUNT = 150;

public:
    IrisDataset(const char*);

    //Sets
    Sample getSample(unsigned int);
    Sample getSet();
    void setType(Sets, bool);
    void closeDataset();

    unsigned classNameToIndex(char *);
    static size_t inputCount();
    static size_t outputCount();
};