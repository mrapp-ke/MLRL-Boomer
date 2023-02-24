#include "boosting/prediction/predictor_binary_label_wise.hpp"

#include "common/iterator/index_iterator.hpp"
#include "common/prediction/predictor_common.hpp"
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

    static inline void applyThreshold(CContiguousConstView<float64>::value_const_iterator originalIterator,
                                      BinaryLilMatrix::row predictionRow, uint32 numElements, float64 threshold) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];

            if (originalValue > threshold) {
                predictionRow.emplace_back(i);
            }
        }
    }

    static inline void predictForExampleInternally(const CContiguousConstView<const float32>& featureMatrix,
                                                   const RuleList& model, CContiguousView<uint8>& predictionMatrix,
                                                   uint32 maxRules, uint32 exampleIndex, float64 threshold) {
        uint32 numLabels = predictionMatrix.getNumCols();
        float64* scoreVector = new float64[numLabels] {};
        applyRules(model, maxRules, featureMatrix.row_values_cbegin(exampleIndex),
                   featureMatrix.row_values_cend(exampleIndex), &scoreVector[0]);
        applyThreshold(&scoreVector[0], predictionMatrix.row_values_begin(exampleIndex), numLabels, threshold);
        delete[] scoreVector;
    }

    static inline void predictForExampleInternally(const CsrConstView<const float32>& featureMatrix,
                                                   const RuleList& model, CContiguousView<uint8>& predictionMatrix,
                                                   uint32 maxRules, uint32 exampleIndex, float64 threshold) {
        uint32 numFeatures = featureMatrix.getNumCols();
        uint32 numLabels = predictionMatrix.getNumCols();
        float64* scoreVector = new float64[numLabels] {};
        applyRules(model, maxRules, numFeatures, featureMatrix.row_indices_cbegin(exampleIndex),
                   featureMatrix.row_indices_cend(exampleIndex), featureMatrix.row_values_cbegin(exampleIndex),
                   featureMatrix.row_values_cend(exampleIndex), &scoreVector[0]);
        applyThreshold(&scoreVector[0], predictionMatrix.row_values_begin(exampleIndex), numLabels, threshold);
        delete[] scoreVector;
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

            typedef PredictionDispatcher<uint8, FeatureMatrix, Model> Dispatcher;

            class Delegate final : public Dispatcher::IPredictionDelegate {
                private:

                    CContiguousView<uint8>& predictionMatrix_;

                    float64 threshold_;

                public:

                    Delegate(CContiguousView<uint8>& predictionMatrix, float64 threshold)
                        : predictionMatrix_(predictionMatrix), threshold_(threshold) {}

                    void predictForExample(const FeatureMatrix& featureMatrix, const Model& model, uint32 maxRules,
                                           uint32 exampleIndex) const override {
                        predictForExampleInternally(featureMatrix, model, predictionMatrix_, maxRules, exampleIndex,
                                                    threshold_);
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

            float64 threshold_;

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
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  threshold_(threshold) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels_);
                Delegate delegate(*predictionMatrixPtr, threshold_);
                Dispatcher().predict(delegate, featureMatrix_, model_, maxRules, numThreads_);
                return predictionMatrixPtr;
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

    static inline void predictForExampleInternally(const CContiguousConstView<const float32>& featureMatrix,
                                                   const RuleList& model, BinaryLilMatrix::row predictionRow,
                                                   uint32 numLabels, uint32 maxRules, uint32 exampleIndex,
                                                   float64 threshold) {
        float64* scoreVector = new float64[numLabels] {};
        applyRules(model, maxRules, featureMatrix.row_values_cbegin(exampleIndex),
                   featureMatrix.row_values_cend(exampleIndex), &scoreVector[0]);
        applyThreshold(&scoreVector[0], predictionRow, numLabels, threshold);
        delete[] scoreVector;
    }

    static inline void predictForExampleInternally(const CsrConstView<const float32>& featureMatrix,
                                                   const RuleList& model, BinaryLilMatrix::row predictionRow,
                                                   uint32 numLabels, uint32 maxRules, uint32 exampleIndex,
                                                   float64 threshold) {
        uint32 numFeatures = featureMatrix.getNumCols();
        float64* scoreVector = new float64[numLabels] {};
        applyRules(model, maxRules, numFeatures, featureMatrix.row_indices_cbegin(exampleIndex),
                   featureMatrix.row_indices_cend(exampleIndex), featureMatrix.row_values_cbegin(exampleIndex),
                   featureMatrix.row_values_cend(exampleIndex), &scoreVector[0]);
        applyThreshold(&scoreVector[0], predictionRow, numLabels, threshold);
        delete[] scoreVector;
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

            typedef BinarySparsePredictionDispatcher<FeatureMatrix, Model> Dispatcher;

            class Delegate final : public Dispatcher::IPredictionDelegate {
                private:

                    BinaryLilMatrix& predictionMatrix_;

                    uint32 numLabels_;

                    float64 threshold_;

                public:

                    Delegate(BinaryLilMatrix& predictionMatrix, uint32 numLabels, float64 threshold)
                        : predictionMatrix_(predictionMatrix), numLabels_(numLabels), threshold_(threshold) {}

                    uint32 predictForExample(const FeatureMatrix& featureMatrix, const Model& model, uint32 maxRules,
                                             uint32 exampleIndex) const override {
                        BinaryLilMatrix::row predictionRow = predictionMatrix_[exampleIndex];
                        predictForExampleInternally(featureMatrix, model, predictionRow, numLabels_, maxRules,
                                                    exampleIndex, threshold_);
                        return (uint32) predictionRow.size();
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

            float64 threshold_;

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
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  threshold_(threshold) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                BinaryLilMatrix predictionMatrix(numExamples);
                Delegate delegate(predictionMatrix, numLabels_, threshold_);
                uint32 numNonZeroElements =
                  Dispatcher().predict(delegate, featureMatrix_, model_, maxRules, numThreads_);
                return createBinarySparsePredictionMatrix(predictionMatrix, numLabels_, numNonZeroElements);
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
