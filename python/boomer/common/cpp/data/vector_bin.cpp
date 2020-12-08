#include "vector_bin.h"


BinVector::BinVector(uint32 numElements)
    : BinVector(numElements, false) {

}

BinVector::BinVector(uint32 numElements, bool init)
    : DenseVector<Bin>(numElements, false) {
    if (init) {
        BinVector::iterator iterator = this->begin();

        for (uint32 i = 0; i < numElements; i++) {
            new (iterator + i) Bin();
        }
    }
}

BinVector::ExampleList& BinVector::getExamples(uint32 binIndex) {
    return examplesPerBin_[binIndex];
}
