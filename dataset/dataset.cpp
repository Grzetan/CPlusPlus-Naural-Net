#include <iostream>
#include <fstream>
#include <assert.h>
#include "dataset.h"
#include "matrix.h"

IrisDataset::IrisDataset(const char* path){
    INPUT_COUNT = 4;
    OUTPUT_COUNT = 3;
    PATH = (char *)path;
    file = NULL;
    TRAIN_COUNT = 80;
    TEST_COUNT = 35;
    VALIDATION_COUNT = 35;
}

Matrix IrisDataset::trainingSet(unsigned int index){
    if(index > TRAIN_COUNT){
        this->indexOutOfRange();
        std::exit(0);
    }
    Matrix c = this->getSample(index);

    Matrix b(1,1,1);
    return b;
}

Matrix IrisDataset::testSet(unsigned int index){
    if(index > TEST_COUNT){
        this->indexOutOfRange();
        std::exit(0);
    }
    index = TRAIN_COUNT + index;

    Matrix b(1,1,1);
    return b;
    // return this->getSample(index);
}

Matrix IrisDataset::validationSet(unsigned int index){
    if(index > VALIDATION_COUNT){
        this->indexOutOfRange();
        std::exit(0);
    }
    index = TRAIN_COUNT + TEST_COUNT + index;

    Matrix b(1,1,1);
    return b;
    // return this->getSample(index);
}

Matrix IrisDataset::getSample(unsigned int seek){
    if(file != NULL){
        char* line = NULL;
        size_t read;
        size_t len = 0;
        while ((read = getline(&line, &len, file)) != -1) {
            std::cout << read << std::endl;
        }   
    }else{
        file = fopen(PATH, "r");
        assert(file != NULL);
        this->getSample(seek);
    }

    Matrix b(1,1,1);
    return b;
}

void IrisDataset::closeDataset(){
    fclose(file);
}

void IrisDataset::indexOutOfRange(){
    try{
        throw std::out_of_range("Provided index is out of range of this set");
    }catch(std::out_of_range error){
        std::cout << error.what() << std::endl;
    }
}