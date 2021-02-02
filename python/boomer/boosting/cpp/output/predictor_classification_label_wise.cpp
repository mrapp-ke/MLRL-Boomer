#include "predictor_classification_label_wise.h"
#include "predictor_common.h"


namespace boosting {

    static inline void applyThreshold(CContiguousView<float64>::const_iterator originalIterator,
                                      CContiguousView<uint8>::iterator transformedIterator, uint32 numElements,
                                      float64 threshold) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];
            transformedIterator[i] = originalValue > threshold ? 1 : 0;
        }
    }

    static inline void predictInternally(const RuleModel& model, const CContiguousFeatureMatrix& featureMatrix,
                                         CContiguousView<float64>& scoreMatrix,
                                         CContiguousView<uint8>& predictionMatrix, float64 threshold) {
        uint32 numExamples = featureMatrix.getNumRows();

        for (uint32 i = 0; i < numExamples; i++) {
            uint32 numFeatures = featureMatrix.getNumCols();

            for (auto it = model.cbegin(); it != model.cend(); it++) {
                const Rule& rule = *it;
                applyRule(rule, featureMatrix.row_cbegin(i), featureMatrix.row_cend(i), scoreMatrix.row_begin(i),
                          scoreMatrix.row_end(i));
            }

            applyThreshold(scoreMatrix.row_cbegin(i), predictionMatrix.row_begin(i), numFeatures, threshold);
        }
    }

    static inline void predictCsrInternally(const RuleModel& model, const CsrFeatureMatrix& featureMatrix,
                                            CContiguousView<float64>& scoreMatrix,
                                            CContiguousView<uint8>& predictionMatrix, float64 threshold) {
        uint32 numExamples = featureMatrix.getNumRows();

        for (uint32 i = 0; i < numExamples; i++) {
            uint32 numFeatures = featureMatrix.getNumCols();
            float32 tmpArray1[numFeatures];
            uint32 tmpArray2[numFeatures] = {};
            uint32 n = 1;

            for (auto it = model.cbegin(); it != model.cend(); it++) {
                const Rule& rule = *it;
                applyRuleCsr(rule, featureMatrix.row_indices_cbegin(i), featureMatrix.row_indices_cend(i),
                             featureMatrix.row_values_cbegin(i), featureMatrix.row_values_cend(i),
                             scoreMatrix.row_begin(i), scoreMatrix.row_end(i), tmpArray1, tmpArray2, n);
                n++;
            }

            applyThreshold(scoreMatrix.row_cbegin(i), predictionMatrix.row_begin(i), numFeatures, threshold);
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
        predictCsrInternally(model, featureMatrix, scoreMatrix, predictionMatrix, threshold_);
    }

}
