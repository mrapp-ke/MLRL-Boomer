#include "boosting/output/predictor_classification_label_wise.hpp"
#include "common/validation.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    static inline void applyThreshold(CContiguousConstView<float64>::const_iterator originalIterator,
                                      uint32 numElements, CContiguousView<uint8>& predictionMatrix, uint32 row,
                                      float64 threshold) {
        CContiguousView<uint8>::iterator transformedIterator = predictionMatrix.row_begin(row);

        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];
            uint8 transformedValue = originalValue > threshold ? 1 : 0;
            transformedIterator[i] = transformedValue;
        }
    }

    static inline void applyThreshold(CContiguousConstView<float64>::const_iterator originalIterator,
                                      uint32 numElements, LilMatrix<uint8>& predictionMatrix, uint32 row,
                                      float64 threshold) {
        LilMatrix<uint8>::Row& rowReference = predictionMatrix.getRow(row);

        for (uint32 i = numElements - 1; i >= 0; i--) {
            float64 originalValue = originalIterator[i];

            if (originalValue > threshold) {
                rowReference.emplace_front(i, 1);
            }
        }
    }

    template<typename PredictionMatrix>
    static void inline predictInternally(const CContiguousFeatureMatrix& featureMatrix,
                                         PredictionMatrix& predictionMatrix, const RuleModel& model,
                                         float64 threshold, uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        PredictionMatrix* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(threshold) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
        num_threads(numThreads)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                applyRule(rule, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i), &scoreVector[0]);
            }

            applyThreshold(&scoreVector[0], numLabels, *predictionMatrixPtr, i, threshold);
        }
    }

    template<typename PredictionMatrix>
    static inline void predictInternally(const CsrFeatureMatrix& featureMatrix, PredictionMatrix& predictionMatrix,
                                         const RuleModel& model, float64 threshold, uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        uint32 numFeatures = featureMatrix.getNumCols();
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        PredictionMatrix* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(numFeatures) \
        firstprivate(threshold) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};
            float32 tmpArray1[numFeatures];
            uint32 tmpArray2[numFeatures] = {};
            uint32 n = 1;

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                applyRuleCsr(rule, featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                             featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                             &scoreVector[0], &tmpArray1[0], &tmpArray2[0], n);
                n++;
            }

            applyThreshold(&scoreVector[0], numLabels, *predictionMatrixPtr, i, threshold);
        }
    }

    LabelWiseClassificationPredictor::LabelWiseClassificationPredictor(float64 threshold, uint32 numThreads)
        : threshold_(threshold), numThreads_(numThreads) {
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    void LabelWiseClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model, const LabelVectorSet* labelVectors) const {
        predictInternally(featureMatrix, predictionMatrix, model, threshold_, numThreads_);
    }

    std::unique_ptr<SparsePredictionMatrix<uint8>> LabelWiseClassificationPredictor::predict(
            const CContiguousFeatureMatrix& featureMatrix, const RuleModel& model,
            const LabelVectorSet* labelVectors) const {
        // TODO
        return nullptr;
    }

    void LabelWiseClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model, const LabelVectorSet* labelVectors) const {
        predictInternally(featureMatrix, predictionMatrix, model, threshold_, numThreads_);
    }

    std::unique_ptr<SparsePredictionMatrix<uint8>> LabelWiseClassificationPredictor::predict(
            const CsrFeatureMatrix& featureMatrix, const RuleModel& model, const LabelVectorSet* labelVectors) const {
        // TODO
        return nullptr;
    }

}
