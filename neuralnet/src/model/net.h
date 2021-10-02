#pragma once
#include "Tensor.h"
#include <vector>

namespace Net{
    struct FeedForwardResult{
        Tensor output;
        Tensor net;
    };

    Tensor tanH(Tensor&);
    Tensor DerivativetanH(Tensor&);
    Tensor weightInit(vector<size_t>, unsigned);
    FeedForwardResult feedForward(Tensor&, Tensor&);
}
