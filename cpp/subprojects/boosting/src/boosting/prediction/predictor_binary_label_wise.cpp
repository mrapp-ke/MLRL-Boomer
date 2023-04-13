#include "boosting/prediction/predictor_binary_label_wise.hpp"

#include "boosting/prediction/discretization_function_score.hpp"
#include "boosting/prediction/predictor_binary_common.hpp"
#include "boosting/prediction/transformation_binary_label_wise.hpp"

namespace boosting {

    /**
     * Allows to create instances of the type `IBinaryPredictor` that allow to predict whether individual labels of
     * given query examples are relevant or irrelevant by discretizing the regression scores or probability estimates
     * that are predicted for each label individually.
     */
    class LabelWiseBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            const std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr_;

            const uint32 numThreads_;

        public:

            /**
             * @param discretizationFunctionFactoryPtr  An unique pointer to an object of type
             *                                          `IDiscretizationFunctionFactory` that allows to create the
             *                                          implementation to be used for discretization
             * @param numThreads                        The number of CPU threads to be used to make predictions for
             *                                          different query examples in parallel. Must be at least 1
             */
            LabelWiseBinaryPredictorFactory(
              std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr, uint32 numThreads)
                : discretizationFunctionFactoryPtr_(std::move(discretizationFunctionFactoryPtr)),
                  numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     const IProbabilityCalibrationModel& probabilityCalibrationModel,
                                                     uint32 numLabels) const override {
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(discretizationFunctionFactoryPtr_->create());
                return std::make_unique<BinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     const IProbabilityCalibrationModel& probabilityCalibrationModel,
                                                     uint32 numLabels) const override {
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(discretizationFunctionFactoryPtr_->create());
                return std::make_unique<BinaryPredictor<CsrConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
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

            const std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr_;

            const uint32 numThreads_;

        public:

            /**
             * @param discretizationFunctionFactoryPtr  An unique pointer to an object of type
             *                                          `IDiscretizationFunctionFactory` that allows to create the
             *                                          implementation to be used for discretization
             * @param numThreads                        The number of CPU threads to be used to make predictions for
             *                                          different query examples in parallel. Must be at least 1
             */
            LabelWiseSparseBinaryPredictorFactory(
              std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr, uint32 numThreads)
                : discretizationFunctionFactoryPtr_(std::move(discretizationFunctionFactoryPtr)),
                  numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(
              const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet, const IProbabilityCalibrationModel& probabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(discretizationFunctionFactoryPtr_->create());
                return std::make_unique<SparseBinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(
              const CsrConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet, const IProbabilityCalibrationModel& probabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(discretizationFunctionFactoryPtr_->create());
                return std::make_unique<SparseBinaryPredictor<CsrConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
            }
    };

    LabelWiseBinaryPredictorConfig::LabelWiseBinaryPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : basedOnProbabilities_(false), lossConfigPtr_(lossConfigPtr),
          multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    bool LabelWiseBinaryPredictorConfig::isBasedOnProbabilities() const {
        return basedOnProbabilities_;
    }

    ILabelWiseBinaryPredictorConfig& LabelWiseBinaryPredictorConfig::setBasedOnProbabilities(
      bool basedOnProbabilities) {
        basedOnProbabilities_ = basedOnProbabilities;
        return *this;
    }

    std::unique_ptr<IBinaryPredictorFactory> LabelWiseBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        float64 threshold = lossConfigPtr_->getDefaultPrediction();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr =
          std::make_unique<ScoreDiscretizationFunctionFactory>(threshold);
        return std::make_unique<LabelWiseBinaryPredictorFactory>(std::move(discretizationFunctionFactoryPtr),
                                                                 numThreads);
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> LabelWiseBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        float64 threshold = lossConfigPtr_->getDefaultPrediction();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr =
          std::make_unique<ScoreDiscretizationFunctionFactory>(threshold);
        return std::make_unique<LabelWiseSparseBinaryPredictorFactory>(std::move(discretizationFunctionFactoryPtr),
                                                                       numThreads);
    }

    bool LabelWiseBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
