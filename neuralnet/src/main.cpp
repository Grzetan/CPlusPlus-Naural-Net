#include <matrix.h>
#include <iostream>
#include <vector>

int main(){
    Matrix a(2,2,3.0);
    Matrix b(2,1,2.0);

    Matrix x = a.yConcatonation(b);
    a.print();
    b.print();
    x.print();
    // M.print();
    // transposed = matrixlib::transpose(matrix);

    // matrixlib::printMatrix(matrix);
    // matrixlib::printMatrix(transposed);
}