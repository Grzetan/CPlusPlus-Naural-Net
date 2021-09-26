#include "assets.h"
#include "Tensor.h"

Tensor output_to_class(Tensor &output){
    if(output.dimCount() != 2){
        std::cout << "Output tensor should have 2 dimensions" << std::endl;
        std::exit(0);
    }

    Tensor result({output.getShape()[0]}, 1);
    unsigned i, j, maxIndex;
    for(i=0; i<output.getShape()[0]; i++){
        maxIndex = 0;
        for(j=0; j<output.getShape()[1]; j++){
            if(output({i,j}) > output({i,maxIndex})){
                maxIndex = j;
            }
        }
        result({i,0}) = maxIndex + 1;
    }
    return result;
}

Tensor class_to_output(Tensor &cls, const unsigned &output_count){
    if(cls.dimCount() != 1){
        std::cout << "Class tensor should have 1 dimension" << std::endl;
        std::exit(0);
    }
    
    Tensor output({cls.getShape()[0], output_count}, 0);
    unsigned i, index;
    for(i=0; i<cls.getShape()[0]; i++){
        index = cls({i}) - 1;
        output({i, index}) = 1;
    }
    return output;
}