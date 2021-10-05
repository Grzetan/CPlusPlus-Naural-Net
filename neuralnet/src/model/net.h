#pragma once
#include "Tensor.h"
#include "assets.h"
#include "dataset.h"
#include <vector>
#include <math.h>

namespace Net{
    struct FeedForwardResult{
        Tensor output;
        Tensor net;
    };

    struct EvaluateErrors{
        double outputError;
        double classificationError;
    };

    struct TrainResult{
        Tensor weights;
        double trainOutputError;
        double trainClassificationError;
        double testOutputError;
        double testClassificationError;
        double validationOutputError;
        double validationClassificationError;
    };

    Tensor tanH(Tensor&);
    Tensor DerivativetanH(Tensor&);
    Tensor weightInit(vector<size_t>, double);
    FeedForwardResult feedForward(Tensor&, Tensor&, Tensor&);
    EvaluateErrors evaluate(Tensor&, Tensor&, Tensor&, Tensor&, Tensor&);
    Tensor backpropagation(Tensor&, Tensor&, double, Tensor&);
    TrainResult train(IrisDataset::Sample&, IrisDataset::Sample&, IrisDataset::Sample&);
}
