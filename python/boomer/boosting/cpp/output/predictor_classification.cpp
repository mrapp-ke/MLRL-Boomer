#include "predictor_classification.h"


static inline void aggregatePredictions(const CContiguousFeatureMatrix& featureMatrix,
                                        CContiguousView<float64> scoreMatrix, const RuleModel& model) {
    uint32 numExamples = featureMatrix.getNumRows();

    for (auto it = model.cbegin(); it != model.cend(); it++) {
        const Rule& rule = *it;
        const IBody& body = rule.getBody();
        const IHead& head = rule.getHead();

        for (uint32 i = 0; i < numExamples; i++) {
            if (body.covers(featureMatrix.row_cbegin(i), featureMatrix.row_cend(i))) {
                head.apply(scoreMatrix.row_begin(i), scoreMatrix.row_end(i));
            }
        }
    }
}

static inline void aggregatePredictions(const CsrFeatureMatrix& featureMatrix, CContiguousView<float64>& scoreMatrix,
                                        const RuleModel& model) {
    uint32 numExamples = featureMatrix.getNumRows();
    uint32 numFeatures = featureMatrix.getNumCols();
    float32 tmpArray1[numFeatures];
    uint32 tmpArray2[numFeatures] = {};
    uint32 n = 1;

    for (auto it = model.cbegin(); it != model.cend(); it++) {
        const Rule& rule = *it;
        const IBody& body = rule.getBody();
        const IHead& head = rule.getHead();

        for (uint32 i = 0; i < numExamples; i++) {
            if (body.covers(featureMatrix.row_indices_cbegin(i), featureMatrix.row_indices_cend(i),
                            featureMatrix.row_values_cbegin(i), featureMatrix.row_values_cend(i), &tmpArray1[0],
                            &tmpArray2[0], n)) {
                head.apply(scoreMatrix.row_begin(i), scoreMatrix.row_end(i));
            }

            n++;
        }
    }
}

template<class I, class O>
static inline void applyThreshold(const CContiguousView<I>& originalMatrix, CContiguousView<O>& transformedMatrix,
                                  I threshold) {
    typename CContiguousView<I>::const_iterator originalIterator = originalMatrix.row_cbegin(0);
    typename CContiguousView<O>::iterator transformedIterator = transformedMatrix.row_begin(0);
    uint32 numElements = originalMatrix.getNumRows() * originalMatrix.getNumCols();

    for (uint32 i = 0; i < numElements; i++) {
        I originalValue = originalIterator[i];
        O transformedValue = originalValue > threshold ? 1 : 0;
        transformedIterator[i] = transformedValue;
    }
}

ClassificationPredictor::ClassificationPredictor(float64 threshold)
    : threshold_(threshold) {

}

void ClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                      CContiguousView<uint8>& predictionMatrix, const RuleModel& model) const {
    uint32 numExamples = predictionMatrix.getNumRows();
    uint32 numLabels = predictionMatrix.getNumCols();
    float64 scores[numExamples * numLabels] = {};
    CContiguousView<float64> scoreMatrix(numExamples, numLabels, &scores[0]);
    aggregatePredictions(featureMatrix, scoreMatrix, model);
    applyThreshold<float64, uint8>(scoreMatrix, predictionMatrix, threshold_);
}

void ClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                      CContiguousView<uint8>& predictionMatrix, const RuleModel& model) const {
    uint32 numExamples = predictionMatrix.getNumRows();
    uint32 numLabels = predictionMatrix.getNumCols();
    float64 scores[numExamples * numLabels] = {};
    CContiguousView<float64> scoreMatrix(numExamples, numLabels, &scores[0]);
    aggregatePredictions(featureMatrix, scoreMatrix, model);
    applyThreshold<float64, uint8>(scoreMatrix, predictionMatrix, threshold_);
}
