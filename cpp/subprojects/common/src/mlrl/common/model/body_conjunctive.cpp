#include "mlrl/common/model/body_conjunctive.hpp"

template<typename Threshold, typename Compare>
ConjunctiveBody::ConditionVector<Threshold, Compare>::ConditionVector(uint32 numConditions)
    : IterableIndexedVectorDecorator<IndexedVectorDecorator<AllocatedVector<uint32>, AllocatedVector<Threshold>>>(
        CompositeVector<AllocatedVector<uint32>, AllocatedVector<Threshold>>(
          AllocatedVector<uint32>(numConditions), AllocatedVector<Threshold>(numConditions))) {}

template<typename Threshold, typename Compare>
bool ConjunctiveBody::ConditionVector<Threshold, Compare>::covers(View<const float32>::const_iterator begin,
                                                                  View<const float32>::const_iterator end) const {
    uint32 numConditions = this->getNumElements();
    typename ConditionVector<Threshold, Compare>::index_const_iterator featureIndexIterator = this->indices_cbegin();
    typename ConditionVector<Threshold, Compare>::value_const_iterator thresholdIterator = this->values_cbegin();

    for (uint32 i = 0; i < numConditions; i++) {
        uint32 featureIndex = featureIndexIterator[i];
        Threshold threshold = thresholdIterator[i];
        Threshold featureValue = (Threshold) begin[featureIndex];

        if (!compare_(featureValue, threshold)) {
            return false;
        }
    }

    return true;
}

template<typename Threshold, typename Compare>
bool ConjunctiveBody::ConditionVector<Threshold, Compare>::covers(View<uint32>::const_iterator indicesBegin,
                                                                  View<uint32>::const_iterator indicesEnd,
                                                                  View<float32>::const_iterator valuesBegin,
                                                                  View<float32>::const_iterator valuesEnd,
                                                                  float32 sparseValue, float32* tmpArray1,
                                                                  uint32* tmpArray2, uint32 n) const {
    uint32 numConditions = this->getNumElements();
    typename ConditionVector<Threshold, Compare>::index_const_iterator featureIndexIterator = this->indices_cbegin();
    typename ConditionVector<Threshold, Compare>::value_const_iterator thresholdIterator = this->values_cbegin();

    for (uint32 i = 0; i < numConditions; i++) {
        uint32 featureIndex = featureIndexIterator[i];
        Threshold threshold = thresholdIterator[i];
        Threshold featureValue = (Threshold) (tmpArray2[featureIndex] == n ? tmpArray1[featureIndex] : sparseValue);

        if (!compare_(featureValue, threshold)) {
            return false;
        }
    }

    return true;
}

ConjunctiveBody::ConjunctiveBody(uint32 numNumericalLeq, uint32 numNumericalGr, uint32 numOrdinalLeq,
                                 uint32 numOrdinalGr, uint32 numNominalEq, uint32 numNominalNeq)
    : numericalLeqVector_(numNumericalLeq), numericalGrVector_(numNumericalGr), ordinalLeqVector_(numOrdinalLeq),
      ordinalGrVector_(numOrdinalGr), nominalEqVector_(numNominalEq), nominalNeqVector_(numNominalNeq) {}

uint32 ConjunctiveBody::getNumNumericalLeq() const {
    return numericalLeqVector_.getNumElements();
}

ConjunctiveBody::numerical_threshold_iterator ConjunctiveBody::numerical_leq_thresholds_begin() {
    return numericalLeqVector_.values_begin();
}

ConjunctiveBody::numerical_threshold_iterator ConjunctiveBody::numerical_leq_thresholds_end() {
    return numericalLeqVector_.values_end();
}

ConjunctiveBody::numerical_threshold_const_iterator ConjunctiveBody::numerical_leq_thresholds_cbegin() const {
    return numericalLeqVector_.values_cbegin();
}

ConjunctiveBody::numerical_threshold_const_iterator ConjunctiveBody::numerical_leq_thresholds_cend() const {
    return numericalLeqVector_.values_cend();
}

ConjunctiveBody::index_iterator ConjunctiveBody::numerical_leq_indices_begin() {
    return numericalLeqVector_.indices_begin();
}

ConjunctiveBody::index_iterator ConjunctiveBody::numerical_leq_indices_end() {
    return numericalLeqVector_.indices_end();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::numerical_leq_indices_cbegin() const {
    return numericalLeqVector_.indices_cbegin();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::numerical_leq_indices_cend() const {
    return numericalLeqVector_.indices_cend();
}

uint32 ConjunctiveBody::getNumNumericalGr() const {
    return numericalGrVector_.getNumElements();
}

ConjunctiveBody::numerical_threshold_iterator ConjunctiveBody::numerical_gr_thresholds_begin() {
    return numericalGrVector_.values_begin();
}

ConjunctiveBody::numerical_threshold_iterator ConjunctiveBody::numerical_gr_thresholds_end() {
    return numericalGrVector_.values_end();
}

ConjunctiveBody::numerical_threshold_const_iterator ConjunctiveBody::numerical_gr_thresholds_cbegin() const {
    return numericalGrVector_.values_cbegin();
}

ConjunctiveBody::numerical_threshold_const_iterator ConjunctiveBody::numerical_gr_thresholds_cend() const {
    return numericalGrVector_.values_cend();
}

ConjunctiveBody::index_iterator ConjunctiveBody::numerical_gr_indices_begin() {
    return numericalGrVector_.indices_begin();
}

ConjunctiveBody::index_iterator ConjunctiveBody::numerical_gr_indices_end() {
    return numericalGrVector_.indices_end();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::numerical_gr_indices_cbegin() const {
    return numericalGrVector_.indices_cbegin();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::numerical_gr_indices_cend() const {
    return numericalGrVector_.indices_cend();
}

uint32 ConjunctiveBody::getNumOrdinalLeq() const {
    return ordinalLeqVector_.getNumElements();
}

ConjunctiveBody::ordinal_threshold_iterator ConjunctiveBody::ordinal_leq_thresholds_begin() {
    return ordinalLeqVector_.values_begin();
}

ConjunctiveBody::ordinal_threshold_iterator ConjunctiveBody::ordinal_leq_thresholds_end() {
    return ordinalLeqVector_.values_end();
}

ConjunctiveBody::ordinal_threshold_const_iterator ConjunctiveBody::ordinal_leq_thresholds_cbegin() const {
    return ordinalLeqVector_.values_cbegin();
}

ConjunctiveBody::ordinal_threshold_const_iterator ConjunctiveBody::ordinal_leq_thresholds_cend() const {
    return ordinalLeqVector_.values_cend();
}

ConjunctiveBody::index_iterator ConjunctiveBody::ordinal_leq_indices_begin() {
    return ordinalLeqVector_.indices_begin();
}

ConjunctiveBody::index_iterator ConjunctiveBody::ordinal_leq_indices_end() {
    return ordinalLeqVector_.indices_end();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::ordinal_leq_indices_cbegin() const {
    return ordinalLeqVector_.indices_cbegin();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::ordinal_leq_indices_cend() const {
    return ordinalLeqVector_.indices_cend();
}

uint32 ConjunctiveBody::getNumOrdinalGr() const {
    return ordinalGrVector_.getNumElements();
}

ConjunctiveBody::ordinal_threshold_iterator ConjunctiveBody::ordinal_gr_thresholds_begin() {
    return ordinalGrVector_.values_begin();
}

ConjunctiveBody::ordinal_threshold_iterator ConjunctiveBody::ordinal_gr_thresholds_end() {
    return ordinalGrVector_.values_end();
}

ConjunctiveBody::ordinal_threshold_const_iterator ConjunctiveBody::ordinal_gr_thresholds_cbegin() const {
    return ordinalGrVector_.values_cbegin();
}

ConjunctiveBody::ordinal_threshold_const_iterator ConjunctiveBody::ordinal_gr_thresholds_cend() const {
    return ordinalGrVector_.values_cend();
}

ConjunctiveBody::index_iterator ConjunctiveBody::ordinal_gr_indices_begin() {
    return ordinalGrVector_.indices_begin();
}

ConjunctiveBody::index_iterator ConjunctiveBody::ordinal_gr_indices_end() {
    return ordinalGrVector_.indices_end();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::ordinal_gr_indices_cbegin() const {
    return ordinalGrVector_.indices_cbegin();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::ordinal_gr_indices_cend() const {
    return ordinalGrVector_.indices_cend();
}

uint32 ConjunctiveBody::getNumNominalEq() const {
    return nominalEqVector_.getNumElements();
}

ConjunctiveBody::nominal_threshold_iterator ConjunctiveBody::nominal_eq_thresholds_begin() {
    return nominalEqVector_.values_begin();
}

ConjunctiveBody::nominal_threshold_iterator ConjunctiveBody::nominal_eq_thresholds_end() {
    return nominalEqVector_.values_end();
}

ConjunctiveBody::nominal_threshold_const_iterator ConjunctiveBody::nominal_eq_thresholds_cbegin() const {
    return nominalEqVector_.values_cbegin();
}

ConjunctiveBody::nominal_threshold_const_iterator ConjunctiveBody::nominal_eq_thresholds_cend() const {
    return nominalEqVector_.values_cend();
}

ConjunctiveBody::index_iterator ConjunctiveBody::nominal_eq_indices_begin() {
    return nominalEqVector_.indices_begin();
}

ConjunctiveBody::index_iterator ConjunctiveBody::nominal_eq_indices_end() {
    return nominalEqVector_.indices_end();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::nominal_eq_indices_cbegin() const {
    return nominalEqVector_.indices_cbegin();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::nominal_eq_indices_cend() const {
    return nominalEqVector_.indices_cend();
}

uint32 ConjunctiveBody::getNumNominalNeq() const {
    return nominalNeqVector_.getNumElements();
}

ConjunctiveBody::nominal_threshold_iterator ConjunctiveBody::nominal_neq_thresholds_begin() {
    return nominalNeqVector_.values_begin();
}

ConjunctiveBody::nominal_threshold_iterator ConjunctiveBody::nominal_neq_thresholds_end() {
    return nominalNeqVector_.values_end();
}

ConjunctiveBody::nominal_threshold_const_iterator ConjunctiveBody::nominal_neq_thresholds_cbegin() const {
    return nominalNeqVector_.values_cbegin();
}

ConjunctiveBody::nominal_threshold_const_iterator ConjunctiveBody::nominal_neq_thresholds_cend() const {
    return nominalNeqVector_.values_cend();
}

ConjunctiveBody::index_iterator ConjunctiveBody::nominal_neq_indices_begin() {
    return nominalNeqVector_.indices_begin();
}

ConjunctiveBody::index_iterator ConjunctiveBody::nominal_neq_indices_end() {
    return nominalNeqVector_.indices_end();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::nominal_neq_indices_cbegin() const {
    return nominalNeqVector_.indices_cbegin();
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::nominal_neq_indices_cend() const {
    return nominalNeqVector_.indices_cend();
}

bool ConjunctiveBody::covers(View<const float32>::const_iterator begin, View<const float32>::const_iterator end) const {
    return numericalLeqVector_.covers(begin, end) && numericalGrVector_.covers(begin, end)
           && ordinalLeqVector_.covers(begin, end) && ordinalGrVector_.covers(begin, end)
           && nominalEqVector_.covers(begin, end) && nominalNeqVector_.covers(begin, end);
}

bool ConjunctiveBody::covers(View<uint32>::const_iterator indicesBegin, View<uint32>::const_iterator indicesEnd,
                             View<float32>::const_iterator valuesBegin, View<float32>::const_iterator valuesEnd,
                             float32 sparseValue, float32* tmpArray1, uint32* tmpArray2, uint32 n) const {
    // Copy dense feature values to the temporary arrays...
    uint32 numDenseElements = valuesEnd - valuesBegin;

    for (uint32 i = 0; i < numDenseElements; i++) {
        uint32 featureIndex = indicesBegin[i];
        float32 featureValue = valuesBegin[i];
        tmpArray1[featureIndex] = featureValue;
        tmpArray2[featureIndex] = n;
    }

    return numericalLeqVector_.covers(indicesBegin, indicesEnd, valuesBegin, valuesEnd, sparseValue, tmpArray1,
                                      tmpArray2, n)
           && numericalGrVector_.covers(indicesBegin, indicesEnd, valuesBegin, valuesEnd, sparseValue, tmpArray1,
                                        tmpArray2, n)
           && ordinalLeqVector_.covers(indicesBegin, indicesEnd, valuesBegin, valuesEnd, sparseValue, tmpArray1,
                                       tmpArray2, n)
           && ordinalGrVector_.covers(indicesBegin, indicesEnd, valuesBegin, valuesEnd, sparseValue, tmpArray1,
                                      tmpArray2, n)
           && nominalEqVector_.covers(indicesBegin, indicesEnd, valuesBegin, valuesEnd, sparseValue, tmpArray1,
                                      tmpArray2, n)
           && nominalNeqVector_.covers(indicesBegin, indicesEnd, valuesBegin, valuesEnd, sparseValue, tmpArray1,
                                       tmpArray2, n);
}

void ConjunctiveBody::visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const {
    conjunctiveBodyVisitor(*this);
}
