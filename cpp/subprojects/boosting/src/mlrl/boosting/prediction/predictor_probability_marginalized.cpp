#include "mlrl/boosting/prediction/predictor_probability_marginalized.hpp"

#include "mlrl/boosting/prediction/predictor_probability_common.hpp"
#include "mlrl/boosting/prediction/transformation_probability_marginalized.hpp"
#include "mlrl/common/prediction/probability_calibration_no.hpp"

namespace boosting {

    template<typename FeatureMatrix, typename Model>
    static inline std::unique_ptr<IProbabilityPredictor> createPredictor(
      const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
      MultiThreadingSettings multiThreadingSettings, const LabelVectorSet* labelVectorSet,
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
      const IJointProbabilityFunctionFactory& jointProbabilityFunctionFactory) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr;

        if (labelVectorSet->getNumLabelVectors() > 0) {
            probabilityTransformationPtr = std::make_unique<MarginalizedProbabilityTransformation>(
              *labelVectorSet, jointProbabilityFunctionFactory.create(marginalProbabilityCalibrationModel,
                                                                      jointProbabilityCalibrationModel));
        }

        return std::make_unique<ProbabilityPredictor<FeatureMatrix, Model>>(
          featureMatrix, model, numLabels, multiThreadingSettings, std::move(probabilityTransformationPtr));
    }

    /**
     * Allows to create instances of the type `IProbabilityPredictor` that allow to predict label-wise probabilities for
     * given query examples by marginalizing over the joint probabilities of known label vectors.
     */
    class MarginalizedProbabilityPredictorFactory final : public IProbabilityPredictorFactory {
        private:

            const std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param jointProbabilityFunctionFactoryPtr    An unique pointer to an object of type
             *                                              `IJointProbabilityFunctionFactory` that allows to create
             *                                              implementations of the function to be used to transform
             *                                              scores that are predicted for an example into joint
             *                                              probabilities
             * @param marginalProbabilityCalibrationModel   A pointer to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` to be used for the
             *                                              calibration of marginal probabilities or a null pointer, if
             *                                              no such model is available
             * @param jointProbabilityCalibrationModel      A pointer to an object of type
             *                                              `IJointProbabilityCalibrationModel` to be used for the
             *                                              calibration of joint probabilities or a null pointer, if no
             *                                              such model is available
             * @param multiThreadingSettings                An object of type `MultiThreadingSettings` that stores the
             *                                              settings to be used for making predictions for different
             *                                              query examples in parallel
             */
            MarginalizedProbabilityPredictorFactory(
              std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel,
              MultiThreadingSettings multiThreadingSettings)
                : jointProbabilityFunctionFactoryPtr_(std::move(jointProbabilityFunctionFactoryPtr)),
                  marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel),
                  jointProbabilityCalibrationModel_(jointProbabilityCalibrationModel),
                  multiThreadingSettings_(multiThreadingSettings) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(
              const CContiguousView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return createPredictor(featureMatrix, model, numLabels, multiThreadingSettings_, labelVectorSet,
                                       marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                                            : marginalProbabilityCalibrationModel,
                                       jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                         : jointProbabilityCalibrationModel,
                                       *jointProbabilityFunctionFactoryPtr_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(
              const CsrView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return createPredictor(featureMatrix, model, numLabels, multiThreadingSettings_, labelVectorSet,
                                       marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                                            : marginalProbabilityCalibrationModel,
                                       jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                         : jointProbabilityCalibrationModel,
                                       *jointProbabilityFunctionFactoryPtr_);
            }
    };

    MarginalizedProbabilityPredictorConfig::MarginalizedProbabilityPredictorConfig(
      ReadableProperty<IClassificationLossConfig> lossConfig,
      ReadableProperty<IMultiThreadingConfig> multiThreadingConfig)
        : lossConfig_(std::move(lossConfig)), multiThreadingConfig_(std::move(multiThreadingConfig)) {}

    bool MarginalizedProbabilityPredictorConfig::isProbabilityCalibrationModelUsed() const {
        return noMarginalProbabilityCalibrationModelPtr_ == nullptr;
    }

    IMarginalizedProbabilityPredictorConfig& MarginalizedProbabilityPredictorConfig::setUseProbabilityCalibrationModel(
      bool useProbabilityCalibrationModel) {
        noMarginalProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        noJointProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        return *this;
    }

    std::unique_ptr<IProbabilityPredictorFactory> MarginalizedProbabilityPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
        std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr =
          lossConfig_.get().createJointProbabilityFunctionFactory();

        if (jointProbabilityFunctionFactoryPtr) {
            MultiThreadingSettings multiThreadingSettings =
              multiThreadingConfig_.get().getSettings(featureMatrix, numOutputs);
            return std::make_unique<MarginalizedProbabilityPredictorFactory>(
              std::move(jointProbabilityFunctionFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(),
              noJointProbabilityCalibrationModelPtr_.get(), multiThreadingSettings);
        } else {
            return nullptr;
        }
    }

    bool MarginalizedProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
