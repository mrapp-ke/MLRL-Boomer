#include "vector_bin.h"


BinVector::BinVector(uint32 numElements)
    : BinVector(numElements, false) {

}

BinVector::BinVector(uint32 numElements, bool init)
    : DenseVector<Bin>(numElements, init) {

}

BinVector::ExampleList& BinVector::getExamples(uint32 binIndex) {
    return examplesPerBin_[binIndex];
}
