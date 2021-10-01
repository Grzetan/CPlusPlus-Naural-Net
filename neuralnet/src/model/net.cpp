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