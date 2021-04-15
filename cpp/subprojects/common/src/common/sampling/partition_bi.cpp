#include "common/sampling/partition_bi.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/thresholds/thresholds_subset.hpp"
#include "common/rule_refinement/refinement.hpp"
#include "common/head_refinement/prediction.hpp"


static inline BinaryDokVector* createDokVector(BiPartition::const_iterator iterator, uint32 numElements) {
    BinaryDokVector* vector = new BinaryDokVector();

    for (uint32 i = 0; i < numElements; i++) {
        uint32 index = iterator[i];
        vector->setValue(index);
    }

    return vector;
}

BiPartition::BiPartition(uint32 numFirst, uint32 numSecond)
    : vector_(DenseVector<uint32>(numFirst + numSecond)), numFirst_(numFirst), firstSet_(nullptr), secondSet_(nullptr) {

}

BiPartition::~BiPartition() {
    delete firstSet_;
    delete secondSet_;
}

BiPartition::iterator BiPartition::first_begin() {
    return vector_.begin();
}

BiPartition::iterator BiPartition::first_end() {
    return &vector_.begin()[numFirst_];
}

BiPartition::const_iterator BiPartition::first_cbegin() const {
    return vector_.cbegin();
}

BiPartition::const_iterator BiPartition::first_cend() const {
    return &vector_.cbegin()[numFirst_];
}

BiPartition::iterator BiPartition::second_begin() {
    return &vector_.begin()[numFirst_];
}

BiPartition::iterator BiPartition::second_end() {
    return vector_.end();
}

BiPartition::const_iterator BiPartition::second_cbegin() const {
    return &vector_.cbegin()[numFirst_];
}

BiPartition::const_iterator BiPartition::second_cend() const {
    return vector_.cend();
}

uint32 BiPartition::getNumFirst() const {
    return numFirst_;
}

uint32 BiPartition::getNumSecond() const {
    return vector_.getNumElements() - numFirst_;
}

uint32 BiPartition::getNumElements() const {
    return vector_.getNumElements();
}

const BinaryDokVector& BiPartition::getFirstSet() {
    if (firstSet_ == nullptr) {
        firstSet_ = createDokVector(this->first_cbegin(), this->getNumFirst());
    }

    return *firstSet_;
}

const BinaryDokVector& BiPartition::getSecondSet() {
    if (secondSet_ == nullptr) {
        secondSet_ = createDokVector(this->second_cbegin(), this->getNumSecond());
    }

    return *secondSet_;
}

std::unique_ptr<IInstanceSubSampling> BiPartition::createInstanceSubSampling(const IInstanceSubSamplingFactory& factory,
                                                                             const ILabelMatrix& labelMatrix) {
    return labelMatrix.createInstanceSubSampling(factory, *this);
}

float64 BiPartition::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const ICoverageState& coverageState,
                                         const AbstractPrediction& head) {
    return coverageState.evaluateOutOfSample(thresholdsSubset, *this, head);
}

void BiPartition::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const ICoverageState& coverageState,
                                        Refinement& refinement) {
    coverageState.recalculatePrediction(thresholdsSubset, *this, refinement);
}
