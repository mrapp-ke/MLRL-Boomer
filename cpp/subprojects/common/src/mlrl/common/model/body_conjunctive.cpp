#include "mlrl/common/model/body_conjunctive.hpp"

ConjunctiveBody::ConjunctiveBody(uint32 numNumericalLeq, uint32 numNumericalGr, uint32 numNominalEq,
                                 uint32 numNominalNeq)
    : numNumericalLeq_(numNumericalLeq), numericalLeqFeatureIndices_(new uint32[numNumericalLeq_]),
      numericalLeqThresholds_(new float32[numNumericalLeq_]), numNumericalGr_(numNumericalGr),
      numericalGrFeatureIndices_(new uint32[numNumericalGr_]), numericalGrThresholds_(new float32[numNumericalGr_]),
      numNominalEq_(numNominalEq), nominalEqFeatureIndices_(new uint32[numNominalEq_]),
      nominalEqThresholds_(new float32[numNominalEq_]), numNominalNeq_(numNominalNeq),
      nominalNeqFeatureIndices_(new uint32[numNominalNeq_]), nominalNeqThresholds_(new float32[numNominalNeq_]) {}

ConjunctiveBody::~ConjunctiveBody() {
    delete[] numericalLeqFeatureIndices_;
    delete[] numericalLeqThresholds_;
    delete[] numericalGrFeatureIndices_;
    delete[] numericalGrThresholds_;
    delete[] nominalEqFeatureIndices_;
    delete[] nominalEqThresholds_;
    delete[] nominalNeqFeatureIndices_;
    delete[] nominalNeqThresholds_;
}

uint32 ConjunctiveBody::getNumNumericalLeq() const {
    return numNumericalLeq_;
}

ConjunctiveBody::threshold_iterator ConjunctiveBody::numerical_leq_thresholds_begin() {
    return numericalLeqThresholds_;
}

ConjunctiveBody::threshold_iterator ConjunctiveBody::numerical_leq_thresholds_end() {
    return &numericalLeqThresholds_[numNumericalLeq_];
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::numerical_leq_thresholds_cbegin() const {
    return numericalLeqThresholds_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::numerical_leq_thresholds_cend() const {
    return &numericalLeqThresholds_[numNumericalLeq_];
}

ConjunctiveBody::index_iterator ConjunctiveBody::numerical_leq_indices_begin() {
    return numericalLeqFeatureIndices_;
}

ConjunctiveBody::index_iterator ConjunctiveBody::numerical_leq_indices_end() {
    return &numericalLeqFeatureIndices_[numNumericalLeq_];
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::numerical_leq_indices_cbegin() const {
    return numericalLeqFeatureIndices_;
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::numerical_leq_indices_cend() const {
    return &numericalLeqFeatureIndices_[numNumericalLeq_];
}

uint32 ConjunctiveBody::getNumNumericalGr() const {
    return numNumericalGr_;
}

ConjunctiveBody::threshold_iterator ConjunctiveBody::numerical_gr_thresholds_begin() {
    return numericalGrThresholds_;
}

ConjunctiveBody::threshold_iterator ConjunctiveBody::numerical_gr_thresholds_end() {
    return &numericalGrThresholds_[numNumericalLeq_];
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::numerical_gr_thresholds_cbegin() const {
    return numericalGrThresholds_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::numerical_gr_thresholds_cend() const {
    return &numericalGrThresholds_[numNumericalLeq_];
}

ConjunctiveBody::index_iterator ConjunctiveBody::numerical_gr_indices_begin() {
    return numericalGrFeatureIndices_;
}

ConjunctiveBody::index_iterator ConjunctiveBody::numerical_gr_indices_end() {
    return &numericalGrFeatureIndices_[numNumericalLeq_];
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::numerical_gr_indices_cbegin() const {
    return numericalGrFeatureIndices_;
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::numerical_gr_indices_cend() const {
    return &numericalGrFeatureIndices_[numNumericalLeq_];
}

uint32 ConjunctiveBody::getNumNominalEq() const {
    return numNominalEq_;
}

ConjunctiveBody::threshold_iterator ConjunctiveBody::nominal_eq_thresholds_begin() {
    return nominalEqThresholds_;
}

ConjunctiveBody::threshold_iterator ConjunctiveBody::nominal_eq_thresholds_end() {
    return &nominalEqThresholds_[numNominalEq_];
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::nominal_eq_thresholds_cbegin() const {
    return nominalEqThresholds_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::nominal_eq_thresholds_cend() const {
    return &nominalEqThresholds_[numNominalEq_];
}

ConjunctiveBody::index_iterator ConjunctiveBody::nominal_eq_indices_begin() {
    return nominalEqFeatureIndices_;
}

ConjunctiveBody::index_iterator ConjunctiveBody::nominal_eq_indices_end() {
    return &nominalEqFeatureIndices_[numNominalEq_];
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::nominal_eq_indices_cbegin() const {
    return nominalEqFeatureIndices_;
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::nominal_eq_indices_cend() const {
    return &nominalEqFeatureIndices_[numNominalEq_];
}

uint32 ConjunctiveBody::getNumNominalNeq() const {
    return numNominalNeq_;
}

ConjunctiveBody::threshold_iterator ConjunctiveBody::nominal_neq_thresholds_begin() {
    return nominalNeqThresholds_;
}

ConjunctiveBody::threshold_iterator ConjunctiveBody::nominal_neq_thresholds_end() {
    return &nominalNeqThresholds_[numNominalNeq_];
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::nominal_neq_thresholds_cbegin() const {
    return nominalNeqThresholds_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::nominal_neq_thresholds_cend() const {
    return &nominalNeqThresholds_[numNominalNeq_];
}

ConjunctiveBody::index_iterator ConjunctiveBody::nominal_neq_indices_begin() {
    return nominalNeqFeatureIndices_;
}

ConjunctiveBody::index_iterator ConjunctiveBody::nominal_neq_indices_end() {
    return &nominalNeqFeatureIndices_[numNominalNeq_];
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::nominal_neq_indices_cbegin() const {
    return nominalNeqFeatureIndices_;
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::nominal_neq_indices_cend() const {
    return &nominalNeqFeatureIndices_[numNominalNeq_];
}

bool ConjunctiveBody::covers(VectorConstView<const float32>::const_iterator begin,
                             VectorConstView<const float32>::const_iterator end) const {
    // Test numerical conditions using the <= operator...
    for (uint32 i = 0; i < numNumericalLeq_; i++) {
        uint32 featureIndex = numericalLeqFeatureIndices_[i];
        float32 threshold = numericalLeqThresholds_[i];

        if (begin[featureIndex] > threshold) {
            return false;
        }
    }

    // Test numerical conditions using the > operator...
    for (uint32 i = 0; i < numNumericalGr_; i++) {
        uint32 featureIndex = numericalGrFeatureIndices_[i];
        float32 threshold = numericalGrThresholds_[i];

        if (begin[featureIndex] <= threshold) {
            return false;
        }
    }

    // Test nominal conditions using the == operator...
    for (uint32 i = 0; i < numNominalEq_; i++) {
        uint32 featureIndex = nominalEqFeatureIndices_[i];
        float32 threshold = nominalEqThresholds_[i];

        if (begin[featureIndex] != threshold) {
            return false;
        }
    }

    // Test nominal conditions using the != operator...
    for (uint32 i = 0; i < numNominalNeq_; i++) {
        uint32 featureIndex = nominalNeqFeatureIndices_[i];
        float32 threshold = nominalNeqThresholds_[i];

        if (begin[featureIndex] == threshold) {
            return false;
        }
    }

    return true;
}

bool ConjunctiveBody::covers(CsrConstView<const float32>::index_const_iterator indicesBegin,
                             CsrConstView<const float32>::index_const_iterator indicesEnd,
                             CsrConstView<const float32>::value_const_iterator valuesBegin,
                             CsrConstView<const float32>::value_const_iterator valuesEnd, float32* tmpArray1,
                             uint32* tmpArray2, uint32 n) const {
    // Copy non-zero feature values to the temporary arrays...
    auto valueIterator = valuesBegin;

    for (auto indexIterator = indicesBegin; indexIterator != indicesEnd; indexIterator++) {
        uint32 featureIndex = *indexIterator;
        float32 featureValue = *valueIterator;
        tmpArray1[featureIndex] = featureValue;
        tmpArray2[featureIndex] = n;
        valueIterator++;
    }

    // Test numerical conditions using the <= operator...
    for (uint32 i = 0; i < numNumericalLeq_; i++) {
        uint32 featureIndex = numericalLeqFeatureIndices_[i];
        float32 threshold = numericalLeqThresholds_[i];
        float32 featureValue = tmpArray2[featureIndex] == n ? tmpArray1[featureIndex] : 0;

        if (featureValue > threshold) {
            return false;
        }
    }

    // Test numerical conditions using the > operator...
    for (uint32 i = 0; i < numNumericalGr_; i++) {
        uint32 featureIndex = numericalGrFeatureIndices_[i];
        float32 threshold = numericalGrThresholds_[i];
        float32 featureValue = tmpArray2[featureIndex] == n ? tmpArray1[featureIndex] : 0;

        if (featureValue <= threshold) {
            return false;
        }
    }

    // Test nominal conditions using the == operator...
    for (uint32 i = 0; i < numNominalEq_; i++) {
        uint32 featureIndex = nominalEqFeatureIndices_[i];
        float32 threshold = nominalEqThresholds_[i];
        float32 featureValue = tmpArray2[featureIndex] == n ? tmpArray1[featureIndex] : 0;

        if (featureValue != threshold) {
            return false;
        }
    }

    // Test nominal conditions using the != operator...
    for (uint32 i = 0; i < numNominalNeq_; i++) {
        uint32 featureIndex = nominalNeqFeatureIndices_[i];
        float32 threshold = nominalNeqThresholds_[i];
        float32 featureValue = tmpArray2[featureIndex] == n ? tmpArray1[featureIndex] : 0;

        if (featureValue == threshold) {
            return false;
        }
    }

    return true;
}

void ConjunctiveBody::visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const {
    conjunctiveBodyVisitor(*this);
}
