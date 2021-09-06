#include <iostream>
#include <vector>
#include "matrix.h"
#include "assets.h"

#define OUTPUT_COUNT 3

int main(){
    Matrix a(5,1,0.0);
    Matrix b(5,1,1.0);

    Matrix x = a.yConcatonation(b);
    
    Matrix xd = output_to_class(x);
    x.print();
    xd.print();
    Matrix out = class_to_output(xd, OUTPUT_COUNT);
    out.print();
}