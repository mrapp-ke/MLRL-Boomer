#include "boosting/output/predictor_classification_example_wise.hpp"
#include "common/validation.hpp"
#include "common/data/arrays.hpp"
#include "predictor_common.hpp"
#include "omp.h"
#include <algorithm>


namespace boosting {

    static inline const LabelVector* findClosestLabelVector(const float64* scoresBegin, const float64* scoresEnd,
                                                            const ISimilarityMeasure& measure,
                                                            const LabelVectorSet* labelVectorSet) {
        const LabelVector* closestLabelVector = nullptr;

        if (labelVectorSet != nullptr) {
            float64 bestScore = 0;
            uint32 bestCount = 0;

            for (auto it = labelVectorSet->cbegin(); it != labelVectorSet->cend(); it++) {
                const auto& entry = *it;
                const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
                uint32 count = entry.second;
                float64 score = measure.measureSimilarity(*labelVectorPtr, scoresBegin, scoresEnd);

                if (closestLabelVector == nullptr || score < bestScore || (score == bestScore && count > bestCount)) {
                    closestLabelVector = labelVectorPtr.get();
                    bestScore = score;
                    bestCount = count;
                }
            }
        }

        return closestLabelVector;
    }

    static inline void predictLabelVector(CContiguousView<uint8>::iterator predictionIterator, uint32 numElements,
                                          const LabelVector* labelVector) {
        setArrayToZeros(predictionIterator, numElements);

        if (labelVector != nullptr) {
            uint32 numIndices = labelVector->getNumElements();
            LabelVector::index_const_iterator indexIterator = labelVector->indices_cbegin();

            for (uint32 i = 0; i < numIndices; i++) {
                uint32 labelIndex = indexIterator[i];
                predictionIterator[labelIndex] = 1;
            }
        }
    }

    static inline uint32 predictLabelVector(BinaryLilMatrix::Row& row, const LabelVector* labelVector) {
        uint32 numNonZeroElements = 0;

        if (labelVector != nullptr) {
            uint32 numIndices = labelVector->getNumElements();
            LabelVector::index_const_iterator indexIterator = labelVector->indices_cbegin();

            if (numIndices > 0) {
                uint32 labelIndex = indexIterator[0];
                row.emplace_front(labelIndex);
                numNonZeroElements++;
                BinaryLilMatrix::Row::iterator it = row.begin();

                for (uint32 i = 1; i < numIndices; i++) {
                    labelIndex = indexIterator[i];
                    it = row.emplace_after(it, labelIndex);
                    numNonZeroElements++;
                }
            }
        }

        return numNonZeroElements;
    }

    /**
     * An implementation of the type `IExampleWiseClassificationPredictor` that allows to predict known label vectors
     * for given query examples by summing up the scores that are provided by an existing rule-based model and comparing
     * the aggregated score vector to the known label vectors according to a certain distance measure. The label vector
     * that is closest to the aggregated score vector is finally predicted.
     *
     * @tparam Model The type of the rule-based model that is used to obtain predictions
     */
    template<typename Model>
    class ExampleWiseClassificationPredictor final : public IClassificationPredictor {

        private:

            const Model& model_;

            const LabelVectorSet* labelVectorSet_;

            std::unique_ptr<ISimilarityMeasure> similarityMeasurePtr_;

            uint32 numThreads_;

        public:

            /**
             * @param model                         A reference to an object of template type `Model` that should be
             *                                      used to obtain predictions
             * @param labelVectorSet                A pointer to an object of type `LabelVectorSet` that stores all
             *                                      known label vectors or a null pointer, if no such set is available
             * @param similarityMeasureFactoryPtr   An unique pointer to an object of type `ISimilarityMeasure` that
             *                                      implements the similarity measure that should be used to quantify
             *                                      the similarity between predictions and known label vectors
             * @param numThreads                    The number of CPU threads to be used to make predictions for
             *                                      different query examples in parallel. Must be at least 1
             */
            ExampleWiseClassificationPredictor(const Model& model, const LabelVectorSet* labelVectorSet,
                                               std::unique_ptr<ISimilarityMeasure> similarityMeasurePtr,
                                               uint32 numThreads)
                : model_(model), labelVectorSet_(labelVectorSet), similarityMeasurePtr_(std::move(similarityMeasurePtr)),
                  numThreads_(numThreads) {

            }

            void predict(const CContiguousFeatureMatrix& featureMatrix,
                         CContiguousView<uint8>& predictionMatrix) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numLabels = predictionMatrix.getNumCols();
                const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
                const Model* modelPtr = &model_;
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;
                const ISimilarityMeasure* similarityMeasureRawPtr = similarityMeasurePtr_.get();

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
                firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) firstprivate(similarityMeasureRawPtr) \
                firstprivate(labelVectorSetPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRules(*modelPtr, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i),
                               &scoreVector[0]);
                    const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0],
                                                                                   &scoreVector[numLabels],
                                                                                   *similarityMeasureRawPtr,
                                                                                   labelVectorSetPtr);
                    predictLabelVector(predictionMatrixPtr->row_begin(i), numLabels, closestLabelVector);
                    delete[] scoreVector;
                }
            }

            void predict(const CsrFeatureMatrix& featureMatrix,
                         CContiguousView<uint8>& predictionMatrix) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                uint32 numLabels = predictionMatrix.getNumCols();
                const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
                const Model* modelPtr = &model_;
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;
                const ISimilarityMeasure* similarityMeasureRawPtr = similarityMeasurePtr_.get();

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
                firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
                firstprivate(similarityMeasureRawPtr) firstprivate(labelVectorSetPtr) schedule(dynamic) \
                num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                  featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                  featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                    const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0],
                                                                                   &scoreVector[numLabels],
                                                                                   *similarityMeasureRawPtr,
                                                                                   labelVectorSetPtr);
                    predictLabelVector(predictionMatrixPtr->row_begin(i), numLabels, closestLabelVector);
                    delete[] scoreVector;
                }
            }

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(const CContiguousFeatureMatrix& featureMatrix,
                                                                        uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                std::unique_ptr<BinaryLilMatrix> lilMatrixPtr = std::make_unique<BinaryLilMatrix>(numExamples);
                const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
                BinaryLilMatrix* predictionMatrixPtr = lilMatrixPtr.get();
                const Model* modelPtr = &model_;
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;
                const ISimilarityMeasure* similarityMeasureRawPtr = similarityMeasurePtr_.get();
                uint32 numNonZeroElements = 0;

                #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                firstprivate(predictionMatrixPtr) firstprivate(similarityMeasureRawPtr) \
                firstprivate(labelVectorSetPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRules(*modelPtr, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i),
                               &scoreVector[0]);
                    const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0],
                                                                                   &scoreVector[numLabels],
                                                                                   *similarityMeasureRawPtr,
                                                                                   labelVectorSetPtr);
                    numNonZeroElements += predictLabelVector(predictionMatrixPtr->getRow(i), closestLabelVector);
                    delete[] scoreVector;
                }

                return std::make_unique<BinarySparsePredictionMatrix>(std::move(lilMatrixPtr), numLabels,
                                                                      numNonZeroElements);
            }

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(const CsrFeatureMatrix& featureMatrix,
                                                                        uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                std::unique_ptr<BinaryLilMatrix> lilMatrixPtr = std::make_unique<BinaryLilMatrix>(numExamples);
                const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
                BinaryLilMatrix* predictionMatrixPtr = lilMatrixPtr.get();
                const Model* modelPtr = &model_;
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;
                const ISimilarityMeasure* similarityMeasureRawPtr = similarityMeasurePtr_.get();
                uint32 numNonZeroElements = 0;

                #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                firstprivate(numFeatures) firstprivate(numLabels) firstprivate(modelPtr) \
                firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) firstprivate(similarityMeasureRawPtr) \
                firstprivate(labelVectorSetPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                  featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                  featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                    const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0],
                                                                                   &scoreVector[numLabels],
                                                                                   *similarityMeasureRawPtr,
                                                                                   labelVectorSetPtr);
                    numNonZeroElements += predictLabelVector(predictionMatrixPtr->getRow(i), closestLabelVector);
                    delete[] scoreVector;
                }

                return std::make_unique<BinarySparsePredictionMatrix>(std::move(lilMatrixPtr), numLabels,
                                                                      numNonZeroElements);
            }

    };

    ExampleWiseClassificationPredictorFactory::ExampleWiseClassificationPredictorFactory(
            std::unique_ptr<ISimilarityMeasureFactory> similarityMeasureFactoryPtr, uint32 numThreads)
        : similarityMeasureFactoryPtr_(std::move(similarityMeasureFactoryPtr)), numThreads_(numThreads) {
        assertNotNull("similarityMeasureFactoryPtr", similarityMeasureFactoryPtr_.get());
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IClassificationPredictor> ExampleWiseClassificationPredictorFactory::create(
            const RuleList& model, const LabelVectorSet* labelVectorSet) const {
        std::unique_ptr<ISimilarityMeasure> similarityMeasurePtr =
            similarityMeasureFactoryPtr_->createSimilarityMeasure();
        return std::make_unique<ExampleWiseClassificationPredictor<RuleList>>(model, labelVectorSet,
                                                                              std::move(similarityMeasurePtr),
                                                                              numThreads_);
    }

}
