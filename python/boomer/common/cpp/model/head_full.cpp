#include "head_full.h"


FullHead::FullHead(const FullPrediction& prediction)
    : numScores_(prediction.getNumElements()), scores_(new float64[numScores_]) {
    FullPrediction::score_const_iterator iterator = prediction.scores_cbegin();

    for (uint32 i = 0; i < numScores_; i++) {
        scores_[i] = iterator[i];
    }
}

FullHead::~FullHead() {
    delete[] scores_;
}

void FullHead::apply(DenseMatrix<float64>::iterator begin, DenseMatrix<float64>::iterator end) const {
    for (uint32 i = 0; i < numScores_; i++) {
        begin[i] += scores_[i];
    }
}

void FullHead::apply(DenseMatrix<float64>::iterator predictionsBegin, DenseMatrix<float64>::iterator predictionsEnd,
                     DenseMatrix<uint8>::iterator maskBegin, DenseMatrix<uint8>::iterator maskEnd) const {
    for (uint32 i = 0; i < numScores_; i++) {
        if (maskBegin[i]) {
            predictionsBegin[i] += scores_[i];
            maskBegin[i] = false;
        }
    }
}
