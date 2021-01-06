#include "head_partial.h"


PartialHead::PartialHead(const PartialPrediction& prediction)
    : numElements_(prediction.getNumElements()), scores_(new float64[numElements_]),
      labelIndices_(new uint32[numElements_]) {
    PartialPrediction::score_const_iterator scoreIterator = prediction.scores_cbegin();
    PartialPrediction::index_const_iterator indexIterator = prediction.indices_cbegin();

    for (uint32 i = 0; i < numElements_; i++) {
        scores_[i] = scoreIterator[i];
        labelIndices_[i] = indexIterator[i];
    }
}

PartialHead::~PartialHead() {
    delete[] scores_;
    delete[] labelIndices_;
}

uint32 PartialHead::getNumElements() const {
    return numElements_;
}

PartialHead::score_const_iterator PartialHead::scores_cbegin() const {
    return scores_;
}

PartialHead::score_const_iterator PartialHead::scores_cend() const {
    return &scores_[numElements_];
}

PartialHead::index_const_iterator PartialHead::indices_cbegin() const {
    return labelIndices_;
}

PartialHead::index_const_iterator PartialHead::indices_cend() const {
    return &labelIndices_[numElements_];
}

void PartialHead::apply(CContiguousView<float64>::iterator begin, CContiguousView<float64>::iterator end) const {
    for (uint32 i = 0; i < numElements_; i++) {
        uint32 labelIndex = labelIndices_[i];
        begin[labelIndex] += scores_[i];
    }
}

void PartialHead::apply(CContiguousView<float64>::iterator predictionsBegin,
                        CContiguousView<float64>::iterator predictionsEnd, PredictionMask::iterator maskBegin,
                        PredictionMask::iterator maskEnd) const {
    for (uint32 i = 0; i < numElements_; i++) {
        uint32 labelIndex = labelIndices_[i];

        if (!maskBegin[labelIndex]) {
            predictionsBegin[labelIndex] += scores_[i];
            maskBegin[labelIndex] = true;
        }
    }
}
