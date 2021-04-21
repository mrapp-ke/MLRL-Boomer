#include "common/input/nominal_feature_mask_dok.hpp"


bool DokNominalFeatureMask::isNominal(uint32 featureIndex) const {
    return vector_[featureIndex];
}

void DokNominalFeatureMask::setNominal(uint32 featureIndex) {
    vector_.set(featureIndex, true);
}
