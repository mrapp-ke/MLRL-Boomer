#include "common/binning/bin_vector.hpp"
#include <iterator>


BinVectorNew::BinVectorNew(uint32 numElements)
    : BinVectorNew(numElements, false) {

}

BinVectorNew::BinVectorNew(uint32 numElements, bool init)
    : DenseVector<Bin>(numElements) {
    if (init) {
        DenseVector<Bin>::iterator iterator = this->begin();

        for (uint32 i = 0; i < numElements; i++) {
            new (iterator + i) Bin();
            iterator[i].index = i;
        }
    }
}
