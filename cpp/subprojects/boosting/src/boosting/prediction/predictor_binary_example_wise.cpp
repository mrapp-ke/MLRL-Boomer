#include "boosting/prediction/predictor_binary_example_wise.hpp"

#include "boosting/prediction/predictor_binary_common.hpp"
#include "boosting/prediction/transformation_binary_example_wise.hpp"
#include "common/data/matrix_dense.hpp"

#include <stdexcept>

namespace boosting {

    /**
     * An implementation of the type `IBinaryPredictor` that allows to predict known label vectors for given query
     * examples by summing up the scores that are provided by an existing rule-based model and comparing the aggregated
     * score vector to the known label vectors according to a certain distance measure. The label vector that is closest
     * to the aggregated score vector is finally predicted.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ExampleWiseBinaryPredictor final : public IBinaryPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

            const LabelVectorSet& labelVectorSet_;

            std::unique_ptr<IBinaryTransformation> binaryTransformationPtr_;

        public:

            /**
             * @param featureMatrix         A reference to an object of template type `FeatureMatrix` that provides
             *                              row-wise access to the feature values of the query examples
             * @param model                 A reference to an object of template type `Model` that should be used to
             *                              obtain predictions
             * @param labelVectorSet        A reference to an object of type `LabelVectorSet` that stores all known
             *                              label vectors
             * @param numLabels             The number of labels to predict for
             * @param distanceMeasurePtr    An unique pointer to an object of type `IDistanceMeasure` that implements
             *                              the distance measure that should be used to calculate the distance between
             *                              predicted scores and known label vectors
             * @param numThreads            The number of CPU threads to be used to make predictions for different query
             *                              examples in parallel. Must be at least 1
             */
            ExampleWiseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model,
                                       const LabelVectorSet& labelVectorSet, uint32 numLabels,
                                       std::unique_ptr<IDistanceMeasure> distanceMeasurePtr, uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  labelVectorSet_(labelVectorSet),
                  binaryTransformationPtr_(
                    std::make_unique<ExampleWiseBinaryTransformation>(labelVectorSet, std::move(distanceMeasurePtr))) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels_, true);

                if (labelVectorSet_.getNumLabelVectors() > 0) {
                    DenseMatrix<float64> scoreMatrix(numThreads_, numLabels_);
                    BinaryPredictionDelegate<FeatureMatrix, Model> delegate(scoreMatrix, *predictionMatrixPtr,
                                                                            *binaryTransformationPtr_);
                    PredictionDispatcher<uint8, FeatureMatrix, Model>().predict(delegate, featureMatrix_, model_,
                                                                                maxRules, numThreads_);
                }

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

    template<typename FeatureMatrix>
    static inline std::unique_ptr<IBinaryPredictor> createExampleWiseBinaryPredictor(
      const FeatureMatrix& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet, uint32 numLabels,
      const IDistanceMeasureFactory& distanceMeasureFactory, uint32 numThreads) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IDistanceMeasure> distanceMeasurePtr = distanceMeasureFactory.createDistanceMeasure();
        return std::make_unique<ExampleWiseBinaryPredictor<FeatureMatrix, RuleList>>(
          featureMatrix, model, *labelVectorSet, numLabels, std::move(distanceMeasurePtr), numThreads);
    }

    /**
     * Allows to create instances of the type `IBinaryPredictor` that allow to predict known label vectors for given
     * query examples by summing up the scores that are provided by an existing rule-based model and comparing the
     * aggregated score vector to the known label vectors according to a certain distance measure. The label vector that
     * is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param distanceMeasureFactoryPtr An unique pointer to an object of type `IDistanceMeasureFactory` that
             *                                  allows to create implementations of the distance measure that should be
             *                                  used to calculate the distance between predicted scores and known label
             *                                  vectors
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            ExampleWiseBinaryPredictorFactory(std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr,
                                              uint32 numThreads)
                : distanceMeasureFactoryPtr_(std::move(distanceMeasureFactoryPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     uint32 numLabels) const override {
                return createExampleWiseBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                        *distanceMeasureFactoryPtr_, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     uint32 numLabels) const override {
                return createExampleWiseBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                        *distanceMeasureFactoryPtr_, numThreads_);
            }
    };

    /**
     * An implementation of the type `ISparseBinaryPredictor` that allows to predict known label vectors for given query
     * examples by summing up the scores that are provided by an existing rule-based model and comparing the aggregated
     * score vector to the known label vectors according to a certain distance measure. The label vector that is closest
     * to the aggregated score vector is finally predicted.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ExampleWiseSparseBinaryPredictor final : public ISparseBinaryPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

            const LabelVectorSet& labelVectorSet_;

            std::unique_ptr<IBinaryTransformation> binaryTransformationPtr_;

        public:

            /**
             * @param featureMatrix         A reference to an object of template type `FeatureMatrix` that provides
             *                              row-wise access to the feature values of the query examples
             * @param model                 A reference to an object of template type `Model` that should be used to
             *                              obtain predictions
             * @param labelVectorSet        A reference to an object of type `LabelVectorSet` that stores all known
             *                              label vectors
             * @param numLabels             The number of labels to predict for
             * @param distanceMeasurePtr    An unique pointer to an object of type `IDistanceMeasure` that implements
             *                              the distance measure that should be used to calculate the distance between
             *                              predicted scores and known label vectors
             * @param numThreads            The number of CPU threads to be used to make predictions for different query
             *                              examples in parallel. Must be at least 1
             */
            ExampleWiseSparseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model,
                                             const LabelVectorSet& labelVectorSet, uint32 numLabels,
                                             std::unique_ptr<IDistanceMeasure> distanceMeasurePtr, uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  labelVectorSet_(labelVectorSet),
                  binaryTransformationPtr_(
                    std::make_unique<ExampleWiseBinaryTransformation>(labelVectorSet, std::move(distanceMeasurePtr))) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                BinaryLilMatrix predictionMatrix(numExamples);
                uint32 numNonZeroElements;

                if (labelVectorSet_.getNumLabelVectors() > 0) {
                    DenseMatrix<float64> scoreMatrix(numThreads_, numLabels_);
                    BinarySparsePredictionDelegate<FeatureMatrix, Model> delegate(scoreMatrix, predictionMatrix,
                                                                                  *binaryTransformationPtr_);
                    numNonZeroElements = BinarySparsePredictionDispatcher<FeatureMatrix, Model>().predict(
                      delegate, featureMatrix_, model_, maxRules, numThreads_);
                } else {
                    numNonZeroElements = 0;
                }

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

    template<typename FeatureMatrix>
    static inline std::unique_ptr<ISparseBinaryPredictor> createExampleWiseSparseBinaryPredictor(
      const FeatureMatrix& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet, uint32 numLabels,
      const IDistanceMeasureFactory& distanceMeasureFactory, uint32 numThreads) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IDistanceMeasure> distanceMeasurePtr = distanceMeasureFactory.createDistanceMeasure();
        return std::make_unique<ExampleWiseSparseBinaryPredictor<FeatureMatrix, RuleList>>(
          featureMatrix, model, *labelVectorSet, numLabels, std::move(distanceMeasurePtr), numThreads);
    }

    /**
     * Allows to create instances of the type `ISparseBinaryPredictor` that allow to predict known label vectors for
     * given query examples by summing up the scores that are provided by an existing rule-based model and comparing the
     * aggregated score vector to the known label vectors according to a certain distance measure. The label vector that
     * is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
        private:

            std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param distanceMeasureFactoryPtr An unique pointer to an object of type `IDistanceMeasureFactory` that
             *                                  allows to create implementations of the distance measure that should be
             *                                  used to calculate the distance between predicted scores and known label
             *                                  vectors
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            ExampleWiseSparseBinaryPredictorFactory(std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr,
                                                    uint32 numThreads)
                : distanceMeasureFactoryPtr_(std::move(distanceMeasureFactoryPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                           const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                           uint32 numLabels) const override {
                return createExampleWiseSparseBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                              *distanceMeasureFactoryPtr_, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                           const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                           uint32 numLabels) const override {
                return createExampleWiseSparseBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                              *distanceMeasureFactoryPtr_, numThreads_);
            }
    };

    ExampleWiseBinaryPredictorConfig::ExampleWiseBinaryPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    std::unique_ptr<IBinaryPredictorFactory> ExampleWiseBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr =
          lossConfigPtr_->createDistanceMeasureFactory();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<ExampleWiseBinaryPredictorFactory>(std::move(distanceMeasureFactoryPtr), numThreads);
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> ExampleWiseBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr =
          lossConfigPtr_->createDistanceMeasureFactory();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<ExampleWiseSparseBinaryPredictorFactory>(std::move(distanceMeasureFactoryPtr),
                                                                         numThreads);
    }

    bool ExampleWiseBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
