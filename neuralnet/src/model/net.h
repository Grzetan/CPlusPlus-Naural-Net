#pragma once
#include "Tensor.h"
#include "assets.h"
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

    Tensor tanH(Tensor&);
    Tensor DerivativetanH(Tensor&);
    Tensor weightInit(vector<size_t>, double);
    FeedForwardResult feedForward(Tensor&, Tensor&, Tensor&);
    EvaluateErrors evaluate(Tensor&, Tensor&, Tensor&, Tensor&, Tensor&);
}
