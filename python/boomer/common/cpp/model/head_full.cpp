#include "head_full.h"


FullHead::FullHead(uint32 numElements)
    : numElements_(numElements), scores_(new float64[numElements]) {

}

FullHead::FullHead(const FullPrediction& prediction)
    : FullHead(prediction.getNumElements()) {
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

void FullHead::visit(FullHeadVisitor fullHeadVisitor, PartialHeadVisitor partialHeadVisitor) const {
    fullHeadVisitor(*this);
}
