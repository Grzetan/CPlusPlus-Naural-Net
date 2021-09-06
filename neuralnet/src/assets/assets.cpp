#include "assets.h"
#include "matrix.h"

Matrix output_to_class(Matrix &output){
    Matrix result(output.getRows(), 1, 0.0);
    unsigned int i, j;
    unsigned int maxIndex;
    for(i=0; i<output.getRows(); i++){
        maxIndex = 0;
        for(j=0; j<output.getCols(); j++){
            if(output(i,j) > output(i,maxIndex)){
                maxIndex = j;
            }
        }
        result(i,0) = maxIndex + 1;
    }
    return result;
}

Matrix class_to_output(Matrix &cls, const unsigned int &output_count){
    Matrix output(cls.getRows(), output_count, 0.0);
    unsigned int i;
    for(i=0; i<cls.getRows(); i++){
        output(i, cls(i,0)-1) = 1;
    }
    return output;
}