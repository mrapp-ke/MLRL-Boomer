#include "predictor_classification_example_wise.h"
#include "predictor_common.h"
#include <algorithm>


namespace boosting {

    template<class T>
    static inline void predictClosestLabelVector(CContiguousView<float64>::const_iterator scoresBegin,
                                                 CContiguousView<float64>::const_iterator scoresEnd,
                                                 CContiguousView<uint8>::iterator predictionsBegin,
                                                 CContiguousView<uint8>::iterator predictionsEnd,
                                                 const IMeasure& measure, const T& labelVectors) {
        std::fill(predictionsBegin, predictionsEnd, 0);
        const LabelVector* closestLabelVector = nullptr;
        float64 bestScore = 0;

        for (auto it = labelVectors.cbegin(); it != labelVectors.cend(); it++) {
            const std::unique_ptr<LabelVector>& labelVectorPtr = *it;
            float64 score = measure.evaluate(*labelVectorPtr, scoresBegin, scoresEnd);

            if (closestLabelVector == nullptr || score < bestScore) {
                closestLabelVector = labelVectorPtr.get();
                bestScore = score;
            }
        }

        if (closestLabelVector != nullptr) {
            for (auto it = closestLabelVector->indices_cbegin(); it != closestLabelVector->indices_cend(); it++) {
                uint32 labelIndex = *it;
                predictionsBegin[labelIndex] = 1;
            }
        }
    }

    template<class T>
    static inline void predictInternally(const RuleModel& model, const CContiguousFeatureMatrix& featureMatrix,
                                         CContiguousView<float64>& scoreMatrix,
                                         CContiguousView<uint8>& predictionMatrix, const IMeasure& measure,
                                         const T& labelVectors) {
        uint32 numExamples = featureMatrix.getNumRows();

        for (uint32 i = 0; i < numExamples; i++) {
            for (auto it = model.cbegin(); it != model.cend(); it++) {
                const Rule& rule = *it;
                applyRule(rule, featureMatrix.row_cbegin(i), featureMatrix.row_cend(i), scoreMatrix.row_begin(i));
            }

            predictClosestLabelVector(scoreMatrix.row_cbegin(i), scoreMatrix.row_cend(i), predictionMatrix.row_begin(i),
                                      predictionMatrix.row_end(i), measure, labelVectors);
        }
    }

    template<class T>
    static inline void predictInternally(const RuleModel& model, const CsrFeatureMatrix& featureMatrix,
                                         CContiguousView<float64>& scoreMatrix,
                                         CContiguousView<uint8>& predictionMatrix, const IMeasure& measure,
                                         const T& labelVectors) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
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

            predictClosestLabelVector(scoreMatrix.row_cbegin(i), scoreMatrix.row_cend(i), predictionMatrix.row_begin(i),
                                      predictionMatrix.row_end(i), measure, labelVectors);
        }
    }

    ExampleWiseClassificationPredictor::ExampleWiseClassificationPredictor(std::shared_ptr<IMeasure> measurePtr)
        : measurePtr_(measurePtr) {

    }

    void ExampleWiseClassificationPredictor::addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr) {
        labelVectors_.emplace(std::move(labelVectorPtr));
    }

    void ExampleWiseClassificationPredictor::visit(LabelVectorVisitor visitor) const {
        for (auto it = labelVectors_.cbegin(); it != labelVectors_.cend(); it++) {
            const std::unique_ptr<LabelVector>& labelVectorPtr = *it;
            visitor(*labelVectorPtr);
        }
    }

    void ExampleWiseClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                     CContiguousView<uint8>& predictionMatrix,
                                                     const RuleModel& model) const {
                                                     uint32 numExamples = predictionMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        float64 scores[numExamples * numLabels] = {};
        CContiguousView<float64> scoreMatrix(numExamples, numLabels, &scores[0]);
        predictInternally<LabelVectorSet>(model, featureMatrix, scoreMatrix, predictionMatrix, *measurePtr_,
                                          labelVectors_);
    }

    void ExampleWiseClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                     CContiguousView<uint8>& predictionMatrix,
                                                     const RuleModel& model) const {
        uint32 numExamples = predictionMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        float64 scores[numExamples * numLabels] = {};
        CContiguousView<float64> scoreMatrix(numExamples, numLabels, &scores[0]);
        predictInternally<LabelVectorSet>(model, featureMatrix, scoreMatrix, predictionMatrix, *measurePtr_,
                                          labelVectors_);
    }

}
