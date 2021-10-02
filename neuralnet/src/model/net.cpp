#include "net.h"
#include <math.h>

Tensor Net::tanH(Tensor& input){
    Tensor output({input.getShape()}, 0);

    for(unsigned i=0; i<input.totalLength; i++){
        double x = exp(2*input.getValues()[i]);
        output(output.linearToComplexIndex(i)) = ((x - 1) / (x + 1) + 1) / 2;
    }
    return output;
}

Tensor Net::DerivativetanH(Tensor& input){
    Tensor output({input.getShape()}, 0);

    for(unsigned i=0; i<input.totalLength; i++){
        double x = exp(2*input.getValues()[i]);
        double tanh = (x - 1) / (x + 1);
        output(output.linearToComplexIndex(i)) = (1 - tanh * tanh) / 2;
    }
    return output;  
}

Tensor Net::weightInit(vector<size_t> shape, unsigned maxWeight){
    Tensor output(shape, 0);
    srand(time(NULL));
    for(unsigned i=0; i<output.totalLength; i++){
        output(output.linearToComplexIndex(i)) = (2*maxWeight) % rand() - maxWeight;
    }
    return output;
}