#include "rule.h"


Rule::Rule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr)
    : bodyPtr_(std::move(bodyPtr)), headPtr_(std::move(headPtr)) {

}

void Rule::predict(const CContiguousFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix) const {
    uint32 numExamples = featureMatrix.getNumExamples();

    for (uint32 r = 0; r < numExamples; r++) {
        bool covers = bodyPtr_->covers(featureMatrix.row_cbegin(r), featureMatrix.row_cend(r));

        if (covers) {
            headPtr_->apply(predictionMatrix.row_begin(r), predictionMatrix.row_end(r));
        }
    }
}

void Rule::predict(const CContiguousFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix,
                   PredictionMask& mask) const {
    uint32 numExamples = featureMatrix.getNumExamples();

    for (uint32 r = 0; r < numExamples; r++) {
        bool covers = bodyPtr_->covers(featureMatrix.row_cbegin(r), featureMatrix.row_cend(r));

        if (covers) {
            headPtr_->apply(predictionMatrix.row_begin(r), predictionMatrix.row_end(r), mask.row_begin(r),
                            mask.row_end(r));
        }
    }
}

void Rule::predict(const CsrFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix, float32* tmpArray1,
                   uint32* tmpArray2, uint32 n) const {
    uint32 numExamples = featureMatrix.getNumExamples();
    uint32 i = n;

    for (uint32 r = 0; r < numExamples; r++) {
        bool covers = bodyPtr_->covers(featureMatrix.row_indices_cbegin(r), featureMatrix.row_indices_cend(r),
                                       featureMatrix.row_values_cbegin(r), featureMatrix.row_values_cend(r), tmpArray1,
                                       tmpArray2, i);

        if (covers) {
            headPtr_->apply(predictionMatrix.row_begin(r), predictionMatrix.row_end(r));
        }

        i++;
    }
}

void Rule::predict(const CsrFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix, float32* tmpArray1,
                   uint32* tmpArray2, uint32 n, PredictionMask& mask) const {
    uint32 numExamples = featureMatrix.getNumExamples();
    uint32 i = n;

    for (uint32 r = 0; r < numExamples; r++) {
        bool covers = bodyPtr_->covers(featureMatrix.row_indices_cbegin(r), featureMatrix.row_indices_cend(r),
                                       featureMatrix.row_values_cbegin(r), featureMatrix.row_values_cend(r), tmpArray1,
                                       tmpArray2, i);

        if (covers) {
            headPtr_->apply(predictionMatrix.row_begin(r), predictionMatrix.row_end(r), mask.row_begin(r),
                            mask.row_end(r));
        }

        i++;
    }
}
