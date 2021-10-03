#include "net.h"

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

Tensor Net::weightInit(vector<size_t> shape, double maxWeight){
    Tensor output(shape, 0);
    srand(time(NULL));
    for(unsigned i=0; i<output.totalLength; i++){
        double f = (double)rand() / RAND_MAX;
        output(output.linearToComplexIndex(i)) = f * (2 * maxWeight) - maxWeight;
    }
    return output;
}

Net::FeedForwardResult Net::feedForward(Tensor& input, Tensor& weights, Tensor& bias){
    Tensor b = input.concat(bias,1);
    Tensor net = b.matmul(weights);
    Tensor output = tanH(net);
    return {net, output};
}

Net::EvaluateErrors Net::evaluate(Tensor& input, Tensor& weights, Tensor& targetOutput, Tensor& targetClass, Tensor& bias){
    unsigned sampleCount = input.getShape()[0];
    FeedForwardResult result = feedForward(input, weights, bias);
    double outputError = (targetOutput - result.output).square().sum() / (sampleCount * targetOutput.getShape()[1]);
    Tensor classes = output_to_class(result.output);
    Tensor clsErrorT({sampleCount}, 0);
    for(unsigned i=0; i<sampleCount; i++){
        if(classes({i}) != targetClass({i})){
            clsErrorT({i}) = 1;
        }
    }
    double clsError = clsErrorT.sum() / sampleCount;
    return {outputError, clsError};
}