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

Tensor Net::backpropagation(Tensor& input, Tensor& weights, double lr, Tensor& bias){
    return weights;
}

Net::TrainResult Net::train(IrisDataset::Sample& trainingSamples, IrisDataset::Sample& testSamples, IrisDataset::Sample& validationSamples){
    bool PLOT = true;
    Tensor weights = Net::weightInit({IrisDataset::inputCount() + 1, IrisDataset::outputCount()}, 0.5);
    Tensor trainBias({trainingSamples.data.getShape()[0], 1}, 1);
    Tensor testBias({testSamples.data.getShape()[0], 1}, 1);
    Tensor validationBias({validationSamples.data.getShape()[0], 1}, 1);

    Tensor error({500, 6},0);
    unsigned i=0;

    while(i<500){
        weights = backpropagation(trainingSamples.data, weights, 0.1, trainBias);

        if(PLOT){
            Tensor targetTrainCls = output_to_class(trainingSamples.labels);
            EvaluateErrors trainErrors = evaluate(trainingSamples.data, weights, trainingSamples.labels, targetTrainCls, trainBias);

            Tensor targetTestCls = output_to_class(testSamples.labels);
            EvaluateErrors testErrors = evaluate(testSamples.data, weights, testSamples.labels, targetTestCls, testBias);
            
            Tensor targetValidationCls = output_to_class(validationSamples.labels);
            EvaluateErrors validationErrors = evaluate(validationSamples.data, weights, validationSamples.labels, targetValidationCls, validationBias);
            error({i, 0}) = trainErrors.outputError;
            error({i, 1}) = trainErrors.classificationError;
            error({i, 2}) = testErrors.outputError;
            error({i, 3}) = testErrors.classificationError;
            error({i, 4}) = validationErrors.outputError;
            error({i, 5}) = validationErrors.classificationError;
        }

        i++;
    }

    if(PLOT){
        //Plot graph
    }

    Tensor targetTrainCls = output_to_class(trainingSamples.labels);
    EvaluateErrors trainErrors = evaluate(trainingSamples.data, weights, trainingSamples.labels, targetTrainCls, trainBias);

    Tensor targetTestCls = output_to_class(testSamples.labels);
    EvaluateErrors testErrors = evaluate(testSamples.data, weights, testSamples.labels, targetTestCls, testBias);
    
    Tensor targetValidationCls = output_to_class(validationSamples.labels);
    EvaluateErrors validationErrors = evaluate(validationSamples.data, weights, validationSamples.labels, targetValidationCls, validationBias);

    return {weights, trainErrors.outputError, trainErrors.classificationError, testErrors.outputError, testErrors.classificationError, validationErrors.outputError, validationErrors.classificationError};
}