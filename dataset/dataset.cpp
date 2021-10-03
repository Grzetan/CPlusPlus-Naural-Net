#include <iostream>
#include <fstream>
#include <assert.h>
#include "dataset.h"
#include "Tensor.h"
#include "string.h"
#include "assets.h"

IrisDataset::IrisDataset(const char* path){
    PATH = (char *)path;
    file = NULL;

    //Shuffle samples
    srand(time(NULL));
    int i, tempIndex;

    for(i=0; i<TOTAL_COUNT; i++){
        indexes[i] = i;
    }
    for(i=TOTAL_COUNT - 1; i>0; i--){
        tempIndex = rand() % i;
        std::swap(indexes[i], indexes[tempIndex]);
    }
}

void IrisDataset::setType(IrisDataset::Sets type, bool _bias = false){
    bias = _bias;
    samples = vector<unsigned>();
    int start, end, i;
    switch(type){
        case TRAIN:
            start = 0;
            end = TRAIN_COUNT;
            samples.reserve(TRAIN_COUNT);
            break;
        case TEST:
            start = TRAIN_COUNT;
            end = TRAIN_COUNT + TEST_COUNT;
            samples.reserve(TEST_COUNT);
            break;
        case VALIDATION:
            start = TRAIN_COUNT + TEST_COUNT;
            end = TRAIN_COUNT + TEST_COUNT + VALIDATION_COUNT;
            samples.reserve(VALIDATION_COUNT);
            break;
    }

    for(i=start; i<end; i++){
        samples.push_back(indexes[i]);
    }
}

IrisDataset::Sample IrisDataset::getSample(unsigned int sample){    
    if(file != NULL){
        if(sample > samples.size()){
            this->indexOutOfRange();
            std::exit(0);
        }
        sample = samples[sample];
        char * line = NULL;
        size_t len = 0;
        unsigned i = 0;
        fseek(file, 0, SEEK_SET);
        while(getline(&line, &len, file)){
            if(i == sample){
                break;
            }
            i++;
        }        
        //Write values to Tensor and extract class
        Tensor data({INPUT_COUNT}, 0);
        Tensor label;
        size_t cls = 0;
        char * splited = strtok(line, ",");
        size_t num = 0;
        while(splited != NULL){
            if(num == INPUT_COUNT){
                cls = this->classNameToIndex(splited);
                Tensor output({OUTPUT_COUNT}, 0);
                output({cls - 1}) = 1;
                label = output;
            }else if(num < INPUT_COUNT){
                data({num}) = atof(splited) / 8;
            }
            num++;
            splited = strtok(NULL, ",");
        }
        return {data, label};
    }else{
        file = fopen(PATH, "r");
        assert(file != NULL);
        return this->getSample(sample);
    }
}

IrisDataset::Sample IrisDataset::getSet(){
    unsigned i;
    Tensor inputs({samples.size(), INPUT_COUNT}, 0);
    Tensor labels({samples.size(), OUTPUT_COUNT}, 0);

    for(i=0; i< samples.size(); i++){
        IrisDataset::Sample sample = this->getSample(i);
        inputs.insertTensor({i},sample.data);
        labels.insertTensor({i},sample.labels);
    }

    return {inputs, labels};
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