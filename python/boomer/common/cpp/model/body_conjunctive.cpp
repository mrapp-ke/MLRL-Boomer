#include "body_conjunctive.h"


uint32 ConjunctiveBody::getNumLeq() const {
    return numLeq_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::leq_values_cbegin() const {
    return leqThresholds_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::leq_values_cend() const {
    return &leqThresholds_[numLeq_];
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::leq_indices_cbegin() const {
    return leqFeatureIndices_;
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::leq_indices_cend() const {
    return &leqFeatureIndices_[numLeq_];
}

uint32 ConjunctiveBody::getNumGr() const {
    return numGr_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::gr_values_cbegin() const {
    return grThresholds_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::gr_values_cend() const {
    return &grThresholds_[numLeq_];
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::gr_indices_cbegin() const {
    return grFeatureIndices_;
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::gr_indices_cend() const {
    return &grFeatureIndices_[numLeq_];
}

uint32 ConjunctiveBody::getNumEq() const {
    return numEq_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::eq_values_cbegin() const {
    return eqThresholds_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::eq_values_cend() const {
    return &eqThresholds_[numLeq_];
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::eq_indices_cbegin() const {
    return eqFeatureIndices_;
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::eq_indices_cend() const {
    return &eqFeatureIndices_[numLeq_];
}

uint32 ConjunctiveBody::getNumNeq() const {
    return numNeq_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::neq_values_cbegin() const {
    return neqThresholds_;
}

ConjunctiveBody::threshold_const_iterator ConjunctiveBody::neq_values_cend() const {
    return &neqThresholds_[numLeq_];
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::neq_indices_cbegin() const {
    return neqFeatureIndices_;
}

ConjunctiveBody::index_const_iterator ConjunctiveBody::neq_indices_cend() const {
    return &neqFeatureIndices_[numLeq_];
}

ConjunctiveBody::ConjunctiveBody(const ConditionList& conditionList)
    : numLeq_(conditionList.getNumConditions(LEQ)), leqFeatureIndices_(new uint32[numLeq_]),
      leqThresholds_(new float32[numLeq_]), numGr_(conditionList.getNumConditions(GR)),
      grFeatureIndices_(new uint32[numGr_]), grThresholds_(new float32[numGr_]),
      numEq_(conditionList.getNumConditions(EQ)), eqFeatureIndices_(new uint32[numEq_]),
      eqThresholds_(new float32[numEq_]), numNeq_(conditionList.getNumConditions(NEQ)),
      neqFeatureIndices_(new uint32[numNeq_]), neqThresholds_(new float32[numNeq_]) {
    uint32 leqIndex = 0;
    uint32 grIndex = 0;
    uint32 eqIndex = 0;
    uint32 neqIndex = 0;

    for (auto it = conditionList.cbegin(); it != conditionList.cend(); it++) {
        const Condition& condition = *it;
        uint32 featureIndex = condition.featureIndex;
        float32 threshold = condition.threshold;

        switch (condition.comparator) {
            case LEQ: {
                leqFeatureIndices_[leqIndex] = featureIndex;
                leqThresholds_[leqIndex] = threshold;
                leqIndex++;
                break;
            }
            case GR: {
                grFeatureIndices_[grIndex] = featureIndex;
                grThresholds_[grIndex] = threshold;
                grIndex++;
                break;
            }
            case EQ: {
                eqFeatureIndices_[eqIndex] = featureIndex;
                eqThresholds_[eqIndex] = threshold;
                eqIndex++;
                break;
            }
            case NEQ: {
                neqFeatureIndices_[neqIndex] = featureIndex;
                neqThresholds_[neqIndex] = threshold;
                neqIndex++;
                break;
            }
            default: { }
        }
    }
}

ConjunctiveBody::~ConjunctiveBody() {
    delete[] leqFeatureIndices_;
    delete[] leqThresholds_;
    delete[] grFeatureIndices_;
    delete[] grThresholds_;
    delete[] eqFeatureIndices_;
    delete[] eqThresholds_;
    delete[] neqFeatureIndices_;
    delete[] neqThresholds_;
}

bool ConjunctiveBody::covers(CContiguousFeatureMatrix::const_iterator begin,
                             CContiguousFeatureMatrix::const_iterator end) const {
    // Test conditions using the <= operator...
    for (uint32 i = 0; i < numLeq_; i++) {
        uint32 featureIndex = leqFeatureIndices_[i];
        float32 threshold = leqThresholds_[i];

        if (begin[featureIndex] > threshold) {
            return false;
        }
    }

    // Test conditions using the > operator...
    for (uint32 i = 0; i < numGr_; i++) {
        uint32 featureIndex = grFeatureIndices_[i];
        float32 threshold = grThresholds_[i];

        if (begin[featureIndex] <= threshold) {
            return false;
        }
    }

    // Test conditions using the == operator...
    for (uint32 i = 0; i < numEq_; i++) {
        uint32 featureIndex = eqFeatureIndices_[i];
        float32 threshold = eqThresholds_[i];

        if (begin[featureIndex] != threshold) {
            return false;
        }
    }

    // Test conditions using the != operator...
    for (uint32 i = 0; i < numNeq_; i++) {
        uint32 featureIndex = neqFeatureIndices_[i];
        float32 threshold = neqThresholds_[i];

        if (begin[featureIndex] == threshold) {
            return false;
        }
    }

    return true;
}

bool ConjunctiveBody::covers(CsrFeatureMatrix::index_const_iterator indicesBegin,
                             CsrFeatureMatrix::index_const_iterator indicesEnd,
                             CsrFeatureMatrix::value_const_iterator valuesBegin,
                             CsrFeatureMatrix::value_const_iterator valuesEnd, float32* tmpArray1, uint32* tmpArray2,
                             uint32 n) const {
    // Copy non-zero feature values to the temporary arrays...
    auto valueIterator = valuesBegin;

    for (auto indexIterator = indicesBegin; indexIterator != indicesEnd; indexIterator++) {
        uint32 featureIndex = *indexIterator;
        float32 featureValue = *valueIterator;
        tmpArray1[featureIndex] = featureValue;
        tmpArray2[featureIndex] = n;
        valueIterator++;
    }

    // Test conditions using the <= operator...
    for (uint32 i = 0; i < numLeq_; i++) {
        uint32 featureIndex = leqFeatureIndices_[i];
        float32 threshold = leqThresholds_[i];
        float32 featureValue = tmpArray2[featureIndex] == n ? tmpArray1[featureIndex] : 0;

        if (threshold > featureValue) {
            return false;
        }
    }

    // Test conditions using the > operator...
    for (uint32 i = 0; i < numGr_; i++) {
        uint32 featureIndex = grFeatureIndices_[i];
        float32 threshold = grThresholds_[i];
        float32 featureValue = tmpArray2[featureIndex] == n ? tmpArray1[featureIndex] : 0;

        if (threshold <= featureValue) {
            return false;
        }
    }

    // Test conditions using the == operator...
    for (uint32 i = 0; i < numEq_; i++) {
        uint32 featureIndex = eqFeatureIndices_[i];
        float32 threshold = eqThresholds_[i];
        float32 featureValue = tmpArray2[featureIndex] == n ? tmpArray1[featureIndex] : 0;

        if (threshold != featureValue) {
            return false;
        }
    }

    // Test conditions using the != operator...
    for (uint32 i = 0; i < numNeq_; i++) {
        uint32 featureIndex = neqFeatureIndices_[i];
        float32 threshold = neqThresholds_[i];
        float32 featureValue = tmpArray2[featureIndex] == n ? tmpArray1[featureIndex] : 0;

        if (threshold == featureValue) {
            return false;
        }
    }

    return true;
}
