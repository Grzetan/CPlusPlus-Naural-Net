#pragma once
#include "Tensor.h"
#include <vector>

namespace Net{
    Tensor tanH(Tensor&);
    Tensor DerivativetanH(Tensor&);
    Tensor weightInit(vector<size_t>, unsigned);
}
