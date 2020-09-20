#include "data.h"


AbstractMatrix::~AbstractMatrix() {

}

uint32 AbstractMatrix::getNumRows() {
    return 0;
}

uint32 AbstractMatrix::getNumCols() {
    return 0;
}

template<class T>
T AbstractRandomAccessMatrix<T>::get(uint32 row, uint32 col) {
    return 0;
}

AbstractVector::~AbstractVector() {

}

uint32 AbstractVector::getNumElements() {
    return 0;
}
