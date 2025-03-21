#include "mlrl/boosting/prediction/predictor_binary_output_wise.hpp"

#include "mlrl/boosting/prediction/discretization_function_probability.hpp"
#include "mlrl/boosting/prediction/discretization_function_score.hpp"
#include "mlrl/boosting/prediction/predictor_binary_common.hpp"
#include "mlrl/boosting/prediction/transformation_binary_output_wise.hpp"
#include "mlrl/common/prediction/probability_calibration_no.hpp"

namespace boosting {

    /**
     * Allows to create instances of the type `IBinaryPredictor` that allow to predict whether individual labels of
     * given query examples are relevant or irrelevant by discretizing the individual scores or probability estimates
     * that are predicted for each label.
     */
    class OutputWiseBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            const std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param discretizationFunctionFactoryPtr      An unique pointer to an object of type
             *                                              `IDiscretizationFunctionFactory` that allows to create the
             *                                              implementation to be used for discretization
             * @param marginalProbabilityCalibrationModel   A pointer to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` to be used for the
             *                                              calibration of marginal probabilities or a null pointer, if
             *                                              no such model is available
             * @param multiThreadingSettings                An object of type `MultiThreadingSettings` that stores the
             *                                              settings to be used for making predictions for different
             *                                              query examples in parallel
             */
            OutputWiseBinaryPredictorFactory(
              std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel,
              MultiThreadingSettings multiThreadingSettings)
                : discretizationFunctionFactoryPtr_(std::move(discretizationFunctionFactoryPtr)),
                  marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel),
                  multiThreadingSettings_(multiThreadingSettings) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(
              const CContiguousView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr =
                  discretizationFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel_
                                                              ? *marginalProbabilityCalibrationModel_
                                                              : marginalProbabilityCalibrationModel);
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<OutputWiseBinaryTransformation>(std::move(discretizationFunctionPtr));
                return std::make_unique<BinaryPredictor<CContiguousView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, multiThreadingSettings_, std::move(binaryTransformationPtr));
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(
              const CsrView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr =
                  discretizationFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel_
                                                              ? *marginalProbabilityCalibrationModel_
                                                              : marginalProbabilityCalibrationModel);
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<OutputWiseBinaryTransformation>(std::move(discretizationFunctionPtr));
                return std::make_unique<BinaryPredictor<CsrView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, multiThreadingSettings_, std::move(binaryTransformationPtr));
            }
    };

    /**
     * Allows to create instances of the type `ISparseBinaryPredictor` that allow to predict whether individual labels
     * of given query examples are relevant or irrelevant by discretizing the scores or probability estimates that are
     * predicted for each label individually.
     */
    class OutputWiseSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
        private:

            const std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param discretizationFunctionFactoryPtr      An unique pointer to an object of type
             *                                              `IDiscretizationFunctionFactory` that allows to create the
             *                                              implementation to be used for discretization
             * @param marginalProbabilityCalibrationModel   A pointer to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` to be used for the
             *                                              calibration of marginal probabilities or a null pointer, if
             *                                              no such model is available
             * @param multiThreadingSettings                An object of type `MultiThreadingSettings` that stores the
             *                                              settings to be used for making predictions for different
             *                                              query examples in parallel
             */
            OutputWiseSparseBinaryPredictorFactory(
              std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel,
              MultiThreadingSettings multiThreadingSettings)
                : discretizationFunctionFactoryPtr_(std::move(discretizationFunctionFactoryPtr)),
                  marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel),
                  multiThreadingSettings_(multiThreadingSettings) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(
              const CContiguousView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr =
                  discretizationFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel_
                                                              ? *marginalProbabilityCalibrationModel_
                                                              : marginalProbabilityCalibrationModel);
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<OutputWiseBinaryTransformation>(std::move(discretizationFunctionPtr));
                return std::make_unique<SparseBinaryPredictor<CContiguousView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, multiThreadingSettings_, std::move(binaryTransformationPtr));
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(
              const CsrView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr =
                  discretizationFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel_
                                                              ? *marginalProbabilityCalibrationModel_
                                                              : marginalProbabilityCalibrationModel);
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<OutputWiseBinaryTransformation>(std::move(discretizationFunctionPtr));
                return std::make_unique<SparseBinaryPredictor<CsrView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, multiThreadingSettings_, std::move(binaryTransformationPtr));
            }
    };

    static inline std::unique_ptr<IDiscretizationFunctionFactory> createDiscretizationFunctionFactory(
      bool basedOnProbabilities, const IClassificationLossConfig& lossConfig) {
        if (basedOnProbabilities) {
            std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactory =
              lossConfig.createMarginalProbabilityFunctionFactory();

            if (marginalProbabilityFunctionFactory) {
                return std::make_unique<ProbabilityDiscretizationFunctionFactory>(
                  std::move(marginalProbabilityFunctionFactory));
            } else {
                return nullptr;
            }
        } else {
            float64 threshold = lossConfig.getDefaultPrediction();
            return std::make_unique<ScoreDiscretizationFunctionFactory>(threshold);
        }
    }

    OutputWiseBinaryPredictorConfig::OutputWiseBinaryPredictorConfig(
      ReadableProperty<IClassificationLossConfig> lossConfig,
      ReadableProperty<IMultiThreadingConfig> multiThreadingConfig)
        : basedOnProbabilities_(false), lossConfig_(lossConfig), multiThreadingConfig_(multiThreadingConfig) {}

    bool OutputWiseBinaryPredictorConfig::isBasedOnProbabilities() const {
        return basedOnProbabilities_;
    }

    IOutputWiseBinaryPredictorConfig& OutputWiseBinaryPredictorConfig::setBasedOnProbabilities(
      bool basedOnProbabilities) {
        basedOnProbabilities_ = basedOnProbabilities;
        return *this;
    }

    bool OutputWiseBinaryPredictorConfig::isProbabilityCalibrationModelUsed() const {
        return noMarginalProbabilityCalibrationModelPtr_ == nullptr;
    }

    IOutputWiseBinaryPredictorConfig& OutputWiseBinaryPredictorConfig::setUseProbabilityCalibrationModel(
      bool useProbabilityCalibrationModel) {
        noMarginalProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        return *this;
    }

    std::unique_ptr<IBinaryPredictorFactory> OutputWiseBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
        std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr =
          createDiscretizationFunctionFactory(basedOnProbabilities_, lossConfig_.get());

        if (discretizationFunctionFactoryPtr) {
            MultiThreadingSettings multiThreadingSettings =
              multiThreadingConfig_.get().getSettings(featureMatrix, numOutputs);
            return std::make_unique<OutputWiseBinaryPredictorFactory>(std::move(discretizationFunctionFactoryPtr),
                                                                      noMarginalProbabilityCalibrationModelPtr_.get(),
                                                                      multiThreadingSettings);
        }

        return nullptr;
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> OutputWiseBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr =
          createDiscretizationFunctionFactory(basedOnProbabilities_, lossConfig_.get());

        if (discretizationFunctionFactoryPtr) {
            MultiThreadingSettings multiThreadingSettings =
              multiThreadingConfig_.get().getSettings(featureMatrix, numLabels);
            return std::make_unique<OutputWiseSparseBinaryPredictorFactory>(
              std::move(discretizationFunctionFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(),
              multiThreadingSettings);
        }

        return nullptr;
    }

    bool OutputWiseBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }
}
