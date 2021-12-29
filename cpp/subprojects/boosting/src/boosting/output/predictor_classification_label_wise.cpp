#include "boosting/output/predictor_classification_label_wise.hpp"
#include "common/validation.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    static inline void applyThreshold(CContiguousConstView<float64>::const_iterator originalIterator,
                                      CContiguousView<uint8>::iterator transformedIterator, uint32 numElements,
                                      float64 threshold) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];
            uint8 transformedValue = originalValue > threshold ? 1 : 0;
            transformedIterator[i] = transformedValue;
        }
    }

    static inline uint32 applyThreshold(CContiguousConstView<float64>::const_iterator originalIterator,
                                        BinaryLilMatrix::Row& row, uint32 numElements, float64 threshold) {
        uint32 numNonZeroElements = 0;
        uint32 i = 0;

        for (; i < numElements; i++) {
            float64 originalValue = originalIterator[i];

            if (originalValue > threshold) {
                row.emplace_front(i);
                numNonZeroElements++;
                break;
            }
        }

        BinaryLilMatrix::Row::iterator it = row.begin();

        for (i = i + 1; i < numElements; i++) {
            float64 originalValue = originalIterator[i];

            if (originalValue > threshold) {
                it = row.emplace_after(it, i);
                numNonZeroElements++;
            }
        }

        return numNonZeroElements;
    }

    /**
     * An implementation of the type `IClassificationPredictor` that allows to predict whether individual labels of
     * given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to a certain threshold
     * that is applied to each label individually (1 if a score exceeds the threshold, i.e., the label is relevant, 0
     * otherwise).
     */
    class LabelWiseClassificationPredictor final : public IClassificationPredictor {

        private:

            float64 threshold_;

            uint32 numThreads_;

        public:

            /**
             * @param threshold     The threshold to be used
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseClassificationPredictor(float64 threshold, uint32 numThreads)
                : threshold_(threshold), numThreads_(numThreads) {

            }

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model, const LabelVectorSet* labelVectors) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numLabels = predictionMatrix.getNumCols();
                const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
                const RuleModel* modelPtr = &model;
                const float64 threshold = threshold_;

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(threshold) \
                firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
                schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRules(*modelPtr, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i),
                               &scoreVector[0]);
                    applyThreshold(&scoreVector[0], predictionMatrixPtr->row_begin(i), numLabels, threshold);
                    delete[] scoreVector;
                }
            }

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model, const LabelVectorSet* labelVectors) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                uint32 numLabels = predictionMatrix.getNumCols();
                const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
                const RuleModel* modelPtr = &model;
                const float64 threshold = threshold_;

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
                firstprivate(threshold) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                  featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                  featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                    applyThreshold(&scoreVector[0], predictionMatrixPtr->row_begin(i), numLabels, threshold);
                    delete[] scoreVector;
                }
            }

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                    const CContiguousFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
                    const LabelVectorSet* labelVectors) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                std::unique_ptr<BinaryLilMatrix> lilMatrixPtr = std::make_unique<BinaryLilMatrix>(numExamples);
                const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
                BinaryLilMatrix* predictionMatrixPtr = lilMatrixPtr.get();
                const RuleModel* modelPtr = &model;
                const float64 threshold = threshold_;
                uint32 numNonZeroElements = 0;

                #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numLabels) \
                firstprivate(threshold) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRules(*modelPtr, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i), &scoreVector[0]);
                    numNonZeroElements += applyThreshold(&scoreVector[0], predictionMatrixPtr->getRow(i), numLabels,
                                                         threshold);
                    delete[] scoreVector;
                }

                return std::make_unique<BinarySparsePredictionMatrix>(std::move(lilMatrixPtr), numLabels, numNonZeroElements);
            }

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                    const CsrFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
                    const LabelVectorSet* labelVectors) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                std::unique_ptr<BinaryLilMatrix> lilMatrixPtr = std::make_unique<BinaryLilMatrix>(numExamples);
                const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
                BinaryLilMatrix* predictionMatrixPtr = lilMatrixPtr.get();
                const RuleModel* modelPtr = &model;
                const float64 threshold = threshold_;
                uint32 numNonZeroElements = 0;

                #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                firstprivate(numFeatures) firstprivate(numLabels) firstprivate(threshold) firstprivate(modelPtr) \
                firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
                num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                  featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                  featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                    numNonZeroElements += applyThreshold(&scoreVector[0], predictionMatrixPtr->getRow(i), numLabels,
                                                         threshold);
                    delete[] scoreVector;
                }

                return std::make_unique<BinarySparsePredictionMatrix>(std::move(lilMatrixPtr), numLabels,
                                                                      numNonZeroElements);
            }

    };

    LabelWiseClassificationPredictorFactory::LabelWiseClassificationPredictorFactory(float64 threshold,
                                                                                     uint32 numThreads)
        : threshold_(threshold), numThreads_(numThreads) {
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IClassificationPredictor> LabelWiseClassificationPredictorFactory::create(
            const RuleModel& model) const {
        return std::make_unique<LabelWiseClassificationPredictor>(threshold_, numThreads_);
    }

}
