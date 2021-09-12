#include <iostream>
#include <fstream>
#include <assert.h>
#include "dataset.h"
#include "Tensor.h"
#include "string.h"
#include "assets.h"

IrisDataset::IrisDataset(const char* path){
    INPUT_COUNT = 4;
    OUTPUT_COUNT = 3;
    PATH = (char *)path;
    file = NULL;
    TRAIN_COUNT = 80;
    TEST_COUNT = 35;
    VALIDATION_COUNT = 35;
}

Tensor IrisDataset::trainingSet(unsigned int index){
    if(index > TRAIN_COUNT){
        this->indexOutOfRange();
        std::exit(0);
    }
    Tensor c = this->getSample(index);

    Tensor b({1,1},1);
    return b;
}

Tensor IrisDataset::testSet(unsigned int index){
    if(index > TEST_COUNT){
        this->indexOutOfRange();
        std::exit(0);
    }
    index = TRAIN_COUNT + index;

    Tensor b({1,1},1);
    return b;
    // return this->getSample(index);
}

Tensor IrisDataset::validationSet(unsigned int index){
    if(index > VALIDATION_COUNT){
        this->indexOutOfRange();
        std::exit(0);
    }
    index = TRAIN_COUNT + TEST_COUNT + index;

    Tensor b({1,1},1);
    return b;
    // return this->getSample(index);
}

Tensor IrisDataset::getSample(unsigned int seek){    
    if(file != NULL){
        fseek(file, seek, SEEK_SET);
        char * line = NULL;
        size_t len = 0;
        getline(&line, &len, file); 

        //Write values to Tensor and extract class
        Tensor input({1, INPUT_COUNT}, 0);
        int cls = 0;
        char * splited = strtok(line, ",");
        size_t num = 0;
        while(splited != NULL){
            if(num >= INPUT_COUNT){
                cls = this->classNameToIndex(splited);
            }else{
                input({0, num}) = atof(splited);
            }
            num++;
            splited = strtok(NULL, ",");
        }

    }else{
        file = fopen(PATH, "r");
        assert(file != NULL);
        this->getSample(seek);
    }

    Tensor b({1,1},1);
    return b;
}

void IrisDataset::closeDataset(){
    fclose(file);
}

int IrisDataset::classNameToIndex(char * str){
    if(strstr(str, "Iris-setosa") != NULL){
        return 1;
    }else if(strstr(str, "Iris-versicolor") != NULL){
        return 2;
    }else if(strstr(str, "Iris-virginica") != NULL){
        return 3;
    }else{
        this->classNotFound();
        std::exit(0);
    }
}

void IrisDataset::indexOutOfRange(){
    try{
        throw std::out_of_range("Provided index is out of range of this set");
    }catch(std::out_of_range error){
        std::cout << error.what() << std::endl;
    }
}

void IrisDataset::classNotFound(){
    try{
        throw std::invalid_argument("Provided class name doesn't match any dataset's classes");
    }catch(std::invalid_argument error){
        std::cout << error.what() << std::endl;
    }  
}