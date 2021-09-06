#pragma once

#include <iostream>
#include <vector>

using std::vector;

class Matrix{
private:
    unsigned rows;
    unsigned cols;
    vector<vector<double> > matrix;

public:
    Matrix(unsigned int, unsigned int, double);
    Matrix(const Matrix &);
    ~Matrix();
    
    //Matrix operations
    Matrix operator+(Matrix &);
    Matrix operator-(Matrix &);
    Matrix operator*(Matrix &);
    Matrix transpose();
    Matrix hadamartMultiplication(Matrix &);
    Matrix kroneckerMultiplication(Matrix &);
    Matrix yConcatonation(Matrix &);

    //Scalar operations
    Matrix operator+(double);
    Matrix operator-(double);
    Matrix operator*(double);
    Matrix operator/(double);

    //Other methods
    double& operator()(const unsigned &, const unsigned &);
    void print();
    unsigned int getRows();
    unsigned int getCols();

    //Exceptions
    void invalidShape();
};