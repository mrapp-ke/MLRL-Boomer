#include "head_full.h"


FullHead::FullHead(const FullPrediction& prediction)
    : numElements_(prediction.getNumElements()), scores_(new float64[numElements_]) {
    FullPrediction::score_const_iterator iterator = prediction.scores_cbegin();

    for (uint32 i = 0; i < numElements_; i++) {
        scores_[i] = iterator[i];
    }
}

FullHead::~FullHead() {
    delete[] scores_;
}

uint32 FullHead::getNumElements() const {
    return numElements_;
}

FullHead::score_const_iterator FullHead::scores_cbegin() const {
    return scores_;
}

FullHead::score_const_iterator FullHead::scores_cend() const {
    return &scores_[numElements_];
}

void FullHead::visit(IHead::FullHeadVisitor fullHeadVisitor, IHead::PartialHeadVisitor partialHeadVisitor) const {
    fullHeadVisitor(*this);
}
