#include "common/binning/threshold_vector.hpp"


ThresholdVector::ThresholdVector(uint32 numElements)
    : ThresholdVector(numElements, false) {

}

ThresholdVector::ThresholdVector(uint32 numElements, bool init)
    : DenseVector<float32>(numElements, init) {

}
