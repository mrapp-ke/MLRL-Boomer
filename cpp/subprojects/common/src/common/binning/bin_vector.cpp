#include "common/binning/bin_vector.hpp"
#include <iterator>


BinVector::BinVector(uint32 numElements)
    : DenseVector<Bin>(numElements) {
    DenseVector<Bin>::iterator iterator = this->begin();

    for (uint32 i = 0; i < numElements; i++) {
        new (iterator + i) Bin();
    }
}
