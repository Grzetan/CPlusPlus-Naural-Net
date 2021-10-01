#pragma once

#include <iostream>
#include <vector>

using std::vector;

class Tensor{
private:
    vector<size_t> shape;
    vector<size_t> strides; 
    vector<double> values;
    size_t multiplyArr(vector<size_t>);
    void invalidShape();
    void invalidIndex();
    void invalidAxis();
    void generateStrides();
    vector<double> copyValuesByStrides();

public:
    size_t totalLength;
    bool contiguous = true;

    Tensor();
    Tensor(vector<size_t>, vector<double>, vector<size_t>, bool);
    Tensor(vector<size_t>, double);

    //Tensor operations
    Tensor operator+(Tensor &);
    Tensor operator-(Tensor &);
    Tensor operator*(Tensor &);
    void insertTensor(vector<size_t>, Tensor &);
    Tensor swapaxes(size_t, size_t);
    Tensor transpose();
    Tensor permute(vector<size_t>);
    Tensor matmul(Tensor &);
    Tensor kron(Tensor &);
    Tensor concat(Tensor &, size_t);
    Tensor flatten();
    Tensor reshape(vector<size_t>);
    Tensor unsquezze();
    Tensor squezze();
    double max();
    double min();

    //Scalar operations
    Tensor operator+(double);
    Tensor operator-(double);
    Tensor operator*(double);
    Tensor operator/(double);

    //Getters and setters
    double& operator()(vector<size_t>);
    Tensor operator[](vector<size_t>);
    // void operator=(Tensor&);
    vector<size_t> getShape();
    size_t dimCount();
    vector<double>& getValues();

    //Other methods
    void printShape();
    void printStrides();
    size_t complexIndexToLinearIndex(vector<size_t>);
    vector<size_t> linearToComplexIndex(size_t);
};