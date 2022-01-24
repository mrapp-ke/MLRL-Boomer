#include "boosting/output/predictor_classification_label_wise.hpp"
#include "common/iterator/index_iterator.hpp"
#include "common/output/label_space_info_no.hpp"
#include "common/util/threads.hpp"
#include "common/util/validation.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    static inline void applyThreshold(CContiguousConstView<float64>::value_const_iterator originalIterator,
                                      CContiguousView<uint8>::value_iterator transformedIterator, uint32 numElements,
                                      float64 threshold) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];
            uint8 transformedValue = originalValue > threshold ? 1 : 0;
            transformedIterator[i] = transformedValue;
        }
    }

    static inline uint32 applyThreshold(CContiguousConstView<float64>::value_const_iterator originalIterator,
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
     *
     * @tparam Model The type of the rule-based model that is used to obtain predictions
     */
    template<typename Model>
    class LabelWiseClassificationPredictor final : public IClassificationPredictor {

        private:

            const Model& model_;

            float64 threshold_;

            uint32 numThreads_;

        public:

            /**
             * @param model         A reference to an object of template type `Model` that should be used to obtain
             *                      predictions
             * @param threshold     The threshold to be used
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseClassificationPredictor(const Model& model, float64 threshold, uint32 numThreads)
                : model_(model), threshold_(threshold), numThreads_(numThreads) {

            }

            std::unique_ptr<DensePredictionMatrix<uint8>> predict(
                    const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels);
                const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                const Model* modelPtr = &model_;
                const float64 threshold = threshold_;

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(threshold) \
                firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
                schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                               &scoreVector[0]);
                    applyThreshold(&scoreVector[0], predictionMatrixRawPtr->row_values_begin(i), numLabels, threshold);
                    delete[] scoreVector;
                }

                return predictionMatrixPtr;
            }

            std::unique_ptr<DensePredictionMatrix<uint8>> predict(const CsrConstView<const float32>& featureMatrix,
                                                                  uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels);
                const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                const Model* modelPtr = &model_;
                const float64 threshold = threshold_;

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
                firstprivate(threshold) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                firstprivate(predictionMatrixRawPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                  featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                  featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                    applyThreshold(&scoreVector[0], predictionMatrixRawPtr->row_values_begin(i), numLabels, threshold);
                    delete[] scoreVector;
                }

                return predictionMatrixPtr;
            }

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                    const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                BinaryLilMatrix lilMatrix(numExamples);
                const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
                const Model* modelPtr = &model_;
                const float64 threshold = threshold_;
                uint32 numNonZeroElements = 0;

                #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                firstprivate(numLabels) firstprivate(threshold) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                               &scoreVector[0]);
                    numNonZeroElements += applyThreshold(&scoreVector[0], predictionMatrixPtr->getRow(i), numLabels,
                                                         threshold);
                    delete[] scoreVector;
                }

                return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
            }

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                    const CsrConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                BinaryLilMatrix lilMatrix(numExamples);
                const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
                const Model* modelPtr = &model_;
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

                return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
            }

    };

    /**
     * Allows to create instances of the type `IClassificationPredictor` that allow to predict whether individual labels
     * of given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to a certain threshold
     * that is applied to each label individually (1 if a score exceeds the threshold, i.e., the label is relevant, 0
     * otherwise).
     */
    class LabelWiseClassificationPredictorFactory final : public IClassificationPredictorFactory {

        private:

            float64 threshold_;

            uint32 numThreads_;

        public:

            /**
             * @param threshold     The threshold that should be used to transform predicted scores into binary
             *                      predictions
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseClassificationPredictorFactory(float64 threshold, uint32 numThreads)
                : threshold_(threshold), numThreads_(numThreads) {

            }

            std::unique_ptr<IClassificationPredictor> create(const RuleList& model,
                                                             const LabelVectorSet* labelVectorSet) const override {
                return std::make_unique<LabelWiseClassificationPredictor<RuleList>>(model, threshold_, numThreads_);
            }

    };

    LabelWiseClassificationPredictorConfig::LabelWiseClassificationPredictorConfig()
        : numThreads_(0) {

    }

    uint32 LabelWiseClassificationPredictorConfig:: getNumThreads() const {
        return numThreads_;
    }

    ILabelWiseClassificationPredictorConfig& LabelWiseClassificationPredictorConfig::setNumThreads(uint32 numThreads) {
        if (numThreads != 0) { assertGreaterOrEqual<uint32>("numThreads", numThreads, 1); }
        numThreads_ = numThreads;
        return *this;
    }

    std::unique_ptr<IClassificationPredictorFactory> LabelWiseClassificationPredictorConfig::createClassificationPredictorFactory() const {
        float64 threshold = 0; // TODO Use correct threshold
        uint32 numThreads = getNumAvailableThreads(numThreads_);
        return std::make_unique<LabelWiseClassificationPredictorFactory>(threshold, numThreads);
    }

    std::unique_ptr<ILabelSpaceInfo> LabelWiseClassificationPredictorConfig::createLabelSpaceInfo(
            const IRowWiseLabelMatrix& labelMatrix) const {
        return createNoLabelSpaceInfo();
    }

}
