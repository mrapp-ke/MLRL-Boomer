#include "mlrl/boosting/prediction/predictor_probability_output_wise.hpp"

#include "mlrl/boosting/prediction/predictor_probability_common.hpp"
#include "mlrl/boosting/prediction/transformation_probability_output_wise.hpp"
#include "mlrl/common/prediction/probability_calibration_no.hpp"

namespace boosting {

    /**
     * Allows to create instances of the type `IProbabilityPredictor` that allow to predict label-wise probabilities for
     * given query examples by transforming the individual scores that are predicted for each label into probabilities.
     */
    class OutputWiseProbabilityPredictorFactory final : public IProbabilityPredictorFactory {
        private:

            const std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const uint32 numThreads_;

        public:

            /**
             * @param marginalProbabilityFunctionFactoryPtr An unique pointer to an object of type
             *                                              `IMarginalProbabilityFunctionFactory` that allows to create
             *                                              implementations of the transformation function to be used to
             *                                              transform scores that are predicted for individual labels
             *                                              into probabilities
             * @param marginalProbabilityCalibrationModel   A pointer to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` to be used for the
             *                                              calibration of marginal probabilities or a null pointer, if
             *                                              no such model is available
             * @param numThreads                            The number of CPU threads to be used to make predictions for
             *                                              different query examples in parallel. Must be at least 1
             */
            OutputWiseProbabilityPredictorFactory(
              std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel, uint32 numThreads)
                : marginalProbabilityFunctionFactoryPtr_(std::move(marginalProbabilityFunctionFactoryPtr)),
                  marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(
              const CContiguousView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr =
                  std::make_unique<OutputWiseProbabilityTransformation>(marginalProbabilityFunctionFactoryPtr_->create(
                    marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                         : marginalProbabilityCalibrationModel));
                return std::make_unique<ProbabilityPredictor<CContiguousView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(probabilityTransformationPtr));
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(
              const CsrView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr =
                  std::make_unique<OutputWiseProbabilityTransformation>(marginalProbabilityFunctionFactoryPtr_->create(
                    marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                         : marginalProbabilityCalibrationModel));
                return std::make_unique<ProbabilityPredictor<CsrView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(probabilityTransformationPtr));
            }
    };

    OutputWiseProbabilityPredictorConfig::OutputWiseProbabilityPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    bool OutputWiseProbabilityPredictorConfig::isProbabilityCalibrationModelUsed() const {
        return noMarginalProbabilityCalibrationModelPtr_ == nullptr;
    }

    IOutputWiseProbabilityPredictorConfig& OutputWiseProbabilityPredictorConfig::setUseProbabilityCalibrationModel(
      bool useProbabilityCalibrationModel) {
        noMarginalProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        return *this;
    }

    std::unique_ptr<IProbabilityPredictorFactory> OutputWiseProbabilityPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
        std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr =
          lossConfigPtr_->createMarginalProbabilityFunctionFactory();

        if (marginalProbabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numOutputs);
            return std::make_unique<OutputWiseProbabilityPredictorFactory>(
              std::move(marginalProbabilityFunctionFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(),
              numThreads);
        } else {
            return nullptr;
        }
    }

    bool OutputWiseProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
