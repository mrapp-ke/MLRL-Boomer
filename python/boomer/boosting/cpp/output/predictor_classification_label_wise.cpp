#include "predictor_classification_label_wise.h"
#include "predictor_common.h"


namespace boosting {

    static inline void applyThreshold(CContiguousView<float64>::const_iterator originalIterator,
                                      CContiguousView<uint8>::iterator transformedIterator, uint32 numElements,
                                      float64 threshold) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];
            uint8 transformedValue = originalValue > threshold ? 1 : 0;
            transformedIterator[i] = transformedValue;
        }
    }

    static inline void predictInternally(const RuleModel& model, const CContiguousFeatureMatrix& featureMatrix,
                                         CContiguousView<float64>& scoreMatrix,
                                         CContiguousView<uint8>& predictionMatrix, float64 threshold) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();

        for (uint32 i = 0; i < numExamples; i++) {
            for (auto it = model.cbegin(); it != model.cend(); it++) {
                const Rule& rule = *it;
                applyRule(rule, featureMatrix.row_cbegin(i), featureMatrix.row_cend(i), scoreMatrix.row_begin(i));
            }

            applyThreshold(scoreMatrix.row_cbegin(i), predictionMatrix.row_begin(i), numLabels, threshold);
        }
    }

    static inline void predictInternally(const RuleModel& model, const CsrFeatureMatrix& featureMatrix,
                                         CContiguousView<float64>& scoreMatrix,
                                         CContiguousView<uint8>& predictionMatrix, float64 threshold) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        uint32 numLabels = predictionMatrix.getNumCols();
        float32 tmpArray1[numFeatures];
        uint32 tmpArray2[numFeatures] = {};
        uint32 n = 1;

        for (uint32 i = 0; i < numExamples; i++) {
            for (auto it = model.cbegin(); it != model.cend(); it++) {
                const Rule& rule = *it;
                applyRuleCsr(rule, featureMatrix.row_indices_cbegin(i), featureMatrix.row_indices_cend(i),
                             featureMatrix.row_values_cbegin(i), featureMatrix.row_values_cend(i),
                             scoreMatrix.row_begin(i), &tmpArray1[0], &tmpArray2[0], n);
                n++;
            }

            applyThreshold(scoreMatrix.row_cbegin(i), predictionMatrix.row_begin(i), numLabels, threshold);
        }
    }

    LabelWiseClassificationPredictor::LabelWiseClassificationPredictor(float64 threshold)
        : threshold_(threshold) {

    }

    void LabelWiseClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model) const {
        uint32 numExamples = predictionMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        float64 scores[numExamples * numLabels] = {};
        CContiguousView<float64> scoreMatrix(numExamples, numLabels, &scores[0]);
        predictInternally(model, featureMatrix, scoreMatrix, predictionMatrix, threshold_);
    }

    void LabelWiseClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model) const {
        uint32 numExamples = predictionMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        float64 scores[numExamples * numLabels] = {};
        CContiguousView<float64> scoreMatrix(numExamples, numLabels, &scores[0]);
        predictInternally(model, featureMatrix, scoreMatrix, predictionMatrix, threshold_);
    }

}
