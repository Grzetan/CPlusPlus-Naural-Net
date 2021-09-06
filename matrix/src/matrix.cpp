#include <iostream>
#include <vector>
#include "matrix.h"

using namespace std;
using std::vector;

Matrix::~Matrix(){

}

Matrix::Matrix(unsigned int rowsCount, unsigned int colsCount, double fill_value){
    rows = rowsCount;
    cols = colsCount;
    matrix = vector<vector<double> >(rows, vector<double>(cols, fill_value));
}

Matrix Matrix::operator+(Matrix &b){
    Matrix sum(rows, cols, 0.0);
    unsigned int i, j;
    if(rows == b.getRows() && cols == b.getCols()){
        for(i=0; i<rows; i++){
            for(j=0; j<cols; j++){
                sum(i,j) = this->matrix[i][j] + b(i,j);
            }
        }
        return sum;
    }else{
        this->invalidShape();
        std::exit(0);
    }
}

Matrix Matrix::operator-(Matrix &b){
    Matrix diff(rows, cols, 0.0);
    unsigned int i, j;
    if(rows == b.getRows() && cols == b.getCols()){
        for(i=0; i<rows; i++){
            for(j=0; j<cols; j++){
                diff(i,j) = this->matrix[i][j] - b(i,j);
            }
        }
        return diff;
    }else{
        this->invalidShape();
        std::exit(0);
    }
}

Matrix Matrix::operator*(Matrix &b){
    Matrix multi(rows, b.getCols(), 0.0);
    unsigned int i, j, k;
    double sum;
    if(cols == b.getRows()){
        for(i=0; i<rows; i++){
            for(j=0; j<b.getCols(); j++){
                sum = 0;
                for(k=0; k<cols; k++){
                    sum = sum + this->matrix[i][k] * b(k,j);
                }
                multi(i,j) = sum;
            }
        }
        return multi;
    }else{
        this->invalidShape();
        std::exit(0);
    }
}

Matrix Matrix::transpose(){
    Matrix transposed(cols, rows, 0.0);
    unsigned int i, j;
    for(i=0; i<cols; i++){
        for(j=0; j<rows; j++){
            transposed(i,j) = this->matrix[j][i];
        }
    }
    return transposed;
}

Matrix Matrix::hadamartMultiplication(Matrix &b){
    Matrix multi(cols, rows, 0.0);
    unsigned int i, j;
    if(rows == b.getRows() && cols == b.getCols()){
        for(i=0; i<rows; i++){
            for(j=0; j<cols; j++){
                multi(i,j) = this->matrix[i][j] * b(i,j);
            }
        }
        return multi;
    }else{
        this->invalidShape();
        std::exit(0);
    }
}

Matrix Matrix::kroneckerMultiplication(Matrix &b){
    Matrix multi(rows*b.getRows(), cols*b.getCols(), 0.0);
    unsigned int i, j, p, q;
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            for(p=0; p<b.getRows(); p++){
                for(q=0; q<b.getCols(); q++){
                    multi(i*b.getRows() + p, j*b.getCols() + q) = this->matrix[i][j] * b(i, j);
                }
            }
        }
    }
    return multi;
}

Matrix Matrix::yConcatonation(Matrix &b){
    Matrix concat(rows, cols + b.getCols(), 0.0);
    unsigned int i, j;
    if(rows == b.getRows()){
        for(i=0; i<rows; i++){
            for(j=0; j<cols; j++){
                concat(i,j) = this->matrix[i][j];
            }
        }
        for(i=0; i<b.getRows(); i++){
            for(j=0; j<b.getCols(); j++){
                concat(i,j+cols) = b(i, j);
            }
        }
        return concat;
    }else{
        this->invalidShape();
        std::exit(0);
    }
}

//Scalar operations
Matrix Matrix::operator+(double scalar){
   Matrix sum(rows, cols, 0.0);
    unsigned int i, j;
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            sum(i,j) = this->matrix[i][j] + scalar;
        }
    }
    return sum;
}

Matrix Matrix::operator-(double scalar){
    Matrix diff(rows, cols, 0.0);
    unsigned int i, j;
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            diff(i,j) = this->matrix[i][j] - scalar;
        }
    }
    return diff;
}

Matrix Matrix::operator*(double scalar){
    Matrix multi(rows, cols, 0.0);
    unsigned int i, j;
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            multi(i,j) = this->matrix[i][j] * scalar;
        }
    }
    return multi;
}

Matrix Matrix::operator/(double scalar){
    Matrix div(rows, cols, 0.0);
    unsigned int i, j;
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            div(i,j) = this->matrix[i][j] / scalar;
        }
    }
    return div;
}

//Other methods
unsigned int Matrix::getRows(){
    return rows;
}

unsigned int Matrix::getCols(){
    return cols;
}

double& Matrix::operator()(const unsigned int &row, const unsigned int &col){
    return this->matrix[row][col];
}

void Matrix::print(){
    std::cout << "Matrix shape: " << rows << "x" << cols << std::endl;
    unsigned int i, j;
    for(i=0; i<rows; i++){
        for(j=0; j < cols; j++){
            std::cout << this->matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

//Exceptions
void Matrix::invalidShape(){
    try{ 
        throw std::length_error("Provided matrixes have invalid shapes, so operation can't be done"); 
    }
    catch(std::length_error str){
        std::cout << str.what() << std::endl; 
    }
}