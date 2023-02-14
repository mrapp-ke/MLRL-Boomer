#include "boosting/prediction/predictor_binary_label_wise.hpp"

#include "common/iterator/index_iterator.hpp"
#include "omp.h"
#include "predictor_common.hpp"

#include <stdexcept>

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
                                        BinaryLilMatrix::row row, uint32 numElements, float64 threshold) {
        uint32 numNonZeroElements = 0;

        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];

            if (originalValue > threshold) {
                row.emplace_back(i);
                numNonZeroElements++;
            }
        }

        return numNonZeroElements;
    }

    static inline std::unique_ptr<DensePredictionMatrix<uint8>> predictInternally(
      const CContiguousConstView<const float32>& featureMatrix, const RuleList& model, uint32 numLabels,
      float64 threshold, uint32 numThreads, uint32 maxRules) {
        uint32 numExamples = featureMatrix.getNumRows();
        std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
          std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels);
        const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
        const RuleList* modelPtr = &model;

#pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(threshold) \
  firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) firstprivate(maxRules) \
    schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            float64* scoreVector = new float64[numLabels] {};
            applyRules(*modelPtr, maxRules, featureMatrixPtr->row_values_cbegin(i),
                       featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
            applyThreshold(&scoreVector[0], predictionMatrixRawPtr->row_values_begin(i), numLabels, threshold);
            delete[] scoreVector;
        }

        return predictionMatrixPtr;
    }

    static inline std::unique_ptr<DensePredictionMatrix<uint8>> predictInternally(
      const CsrConstView<const float32>& featureMatrix, const RuleList& model, uint32 numLabels, float64 threshold,
      uint32 numThreads, uint32 maxRules) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
          std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels);
        const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
        const RuleList* modelPtr = &model;

#pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
  firstprivate(threshold) firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
    firstprivate(maxRules) schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            float64* scoreVector = new float64[numLabels] {};
            applyRulesCsr(*modelPtr, maxRules, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                          featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                          featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
            applyThreshold(&scoreVector[0], predictionMatrixRawPtr->row_values_begin(i), numLabels, threshold);
            delete[] scoreVector;
        }

        return predictionMatrixPtr;
    }

    /**
     * An implementation of the type `IBinaryPredictor` that allows to predict whether individual labels of given query
     * examples are relevant or irrelevant by summing up the scores that are provided by the individual rules of an
     * existing rule-based model and transforming them into binary values according to a certain threshold that is
     * applied to each label individually (1 if a score exceeds the threshold, i.e., the label is relevant, 0
     * otherwise).
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class LabelWiseBinaryPredictor final : public IBinaryPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            float64 threshold_;

            uint32 numThreads_;

        public:

            /**
             * @param featureMatrix A reference to an object of template type `FeatureMatrix` that provides row-wise
             *                      access to the feature values of the query examples
             * @param model         A reference to an object of template type `Model` that should be used to obtain
             *                      predictions
             * @param numLabels     The number of labels to predict for
             * @param threshold     The threshold to be used
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                     float64 threshold, uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), threshold_(threshold),
                  numThreads_(numThreads) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(uint32 maxRules) const override {
                return predictInternally(featureMatrix_, model_, numLabels_, threshold_, numThreads_, maxRules);
            }

            /**
             * @see `IPredictor::canPredictIncrementally`
             */
            bool canPredictIncrementally() const override {
                return false;
            }

            /**
             * @see `IPredictor::createIncrementalPredictor`
             */
            std::unique_ptr<IIncrementalPredictor<DensePredictionMatrix<uint8>>> createIncrementalPredictor(
              uint32 minRules, uint32 maxRules) const override {
                throw std::runtime_error("The rule learner does not support to predict binary labels incrementally");
            }
    };

    /**
     * Allows to create instances of the type `IBinaryPredictor` that allow to predict whether individual labels of
     * given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to a certain threshold
     * that is applied to each label individually (1 if a score exceeds the threshold, i.e., the label is relevant, 0
     * otherwise).
     */
    class LabelWiseBinaryPredictorFactory final : public IBinaryPredictorFactory {
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
            LabelWiseBinaryPredictorFactory(float64 threshold, uint32 numThreads)
                : threshold_(threshold), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     uint32 numLabels) const override {
                return std::make_unique<LabelWiseBinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, threshold_, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     uint32 numLabels) const override {
                return std::make_unique<LabelWiseBinaryPredictor<CsrConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, threshold_, numThreads_);
            }
    };

    std::unique_ptr<BinarySparsePredictionMatrix> predictSparseInternally(
      const CContiguousConstView<const float32>& featureMatrix, const RuleList& model, uint32 numLabels,
      float64 threshold, uint32 numThreads, uint32 maxRules) {
        uint32 numExamples = featureMatrix.getNumRows();
        BinaryLilMatrix lilMatrix(numExamples);
        const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
        BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
        const RuleList* modelPtr = &model;
        uint32 numNonZeroElements = 0;

#pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numLabels) \
  firstprivate(threshold) firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
     firstprivate(maxRules) schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            float64* scoreVector = new float64[numLabels] {};
            applyRules(*modelPtr, maxRules, featureMatrixPtr->row_values_cbegin(i),
                       featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
            numNonZeroElements += applyThreshold(&scoreVector[0], (*predictionMatrixPtr)[i], numLabels, threshold);
            delete[] scoreVector;
        }

        return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
    }

    std::unique_ptr<BinarySparsePredictionMatrix> predictSparseInternally(
      const CsrConstView<const float32>& featureMatrix, const RuleList& model, uint32 numLabels, float64 threshold,
      uint32 numThreads, uint32 maxRules) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        BinaryLilMatrix lilMatrix(numExamples);
        const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
        BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
        const RuleList* modelPtr = &model;
        uint32 numNonZeroElements = 0;

#pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numFeatures) \
  firstprivate(numLabels) firstprivate(threshold) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
    firstprivate(predictionMatrixPtr) firstprivate(maxRules) schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            float64* scoreVector = new float64[numLabels] {};
            applyRulesCsr(*modelPtr, maxRules, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                          featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                          featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
            numNonZeroElements += applyThreshold(&scoreVector[0], (*predictionMatrixPtr)[i], numLabels, threshold);
            delete[] scoreVector;
        }

        return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
    }

    /**
     * An implementation of the type `ISparseBinaryPredictor` that allows to predict whether individual labels of given
     * query examples are relevant or irrelevant by summing up the scores that are provided by the individual rules of
     * an existing rule-based model and transforming them into binary values according to a certain threshold that is
     * applied to each label individually (1 if a score exceeds the threshold, i.e., the label is relevant, 0
     * otherwise).
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class LabelWiseSparseBinaryPredictor final : public ISparseBinaryPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            float64 threshold_;

            uint32 numThreads_;

        public:

            /**
             * @param featureMatrix A reference to an object of template type `FeatureMatrix` that provides row-wise
             *                      access to the feature values of the query examples
             * @param model         A reference to an object of template type `Model` that should be used to obtain
             *                      predictions
             * @param numLabels     The number of labels to predict for
             * @param threshold     The threshold to be used
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseSparseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                           float64 threshold, uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), threshold_(threshold),
                  numThreads_(numThreads) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict(uint32 maxRules) const override {
                return predictSparseInternally(featureMatrix_, model_, numLabels_, threshold_, numThreads_, maxRules);
            }

            /**
             * @see `IPredictor::canPredictIncrementally`
             */
            bool canPredictIncrementally() const override {
                return false;
            }

            /**
             * @see `IPredictor::createIncrementalPredictor`
             */
            std::unique_ptr<IIncrementalPredictor<BinarySparsePredictionMatrix>> createIncrementalPredictor(
              uint32 minRules, uint32 maxRules) const override {
                throw std::runtime_error(
                  "The rule learner does not support to predict sparse binary labels incrementally");
            }
    };

    /**
     * Allows to create instances of the type `ISparseBinaryPredictor` that allow to predict whether individual labels
     * of given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to a certain threshold
     * that is applied to each label individually (1 if a score exceeds the threshold, i.e., the label is relevant, 0
     * otherwise).
     */
    class LabelWiseSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
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
            LabelWiseSparseBinaryPredictorFactory(float64 threshold, uint32 numThreads)
                : threshold_(threshold), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                           const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                           uint32 numLabels) const override {
                return std::make_unique<LabelWiseSparseBinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, threshold_, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                           const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                           uint32 numLabels) const override {
                return std::make_unique<LabelWiseSparseBinaryPredictor<CsrConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, threshold_, numThreads_);
            }
    };

    LabelWiseBinaryPredictorConfig::LabelWiseBinaryPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    std::unique_ptr<IBinaryPredictorFactory> LabelWiseBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        float64 threshold = lossConfigPtr_->getDefaultPrediction();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<LabelWiseBinaryPredictorFactory>(threshold, numThreads);
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> LabelWiseBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        float64 threshold = lossConfigPtr_->getDefaultPrediction();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<LabelWiseSparseBinaryPredictorFactory>(threshold, numThreads);
    }

    bool LabelWiseBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
