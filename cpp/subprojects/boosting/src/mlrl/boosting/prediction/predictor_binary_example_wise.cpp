#include "mlrl/boosting/prediction/predictor_binary_example_wise.hpp"

#include "mlrl/boosting/prediction/predictor_binary_common.hpp"
#include "mlrl/boosting/prediction/transformation_binary_example_wise.hpp"
#include "mlrl/common/prediction/probability_calibration_no.hpp"

namespace boosting {

    static inline std::unique_ptr<IBinaryTransformation> createBinaryTransformation(
      const LabelVectorSet* labelVectorSet, const IDistanceMeasureFactory<float64>& distanceMeasureFactory,
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IBinaryTransformation> binaryTransformationPtr;

        if (labelVectorSet->getNumLabelVectors() > 0) {
            binaryTransformationPtr = std::make_unique<ExampleWiseBinaryTransformation>(
              *labelVectorSet, distanceMeasureFactory.createDistanceMeasure(marginalProbabilityCalibrationModel,
                                                                            jointProbabilityCalibrationModel));
        }

        return binaryTransformationPtr;
    }

    template<typename FeatureMatrix, typename Model>
    static inline std::unique_ptr<IBinaryPredictor> createPredictor(
      const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
      MultiThreadingSettings multiThreadingSettings, const LabelVectorSet* labelVectorSet,
      const IDistanceMeasureFactory<float64>& distanceMeasureFactory,
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) {
        std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
          createBinaryTransformation(labelVectorSet, distanceMeasureFactory, marginalProbabilityCalibrationModel,
                                     jointProbabilityCalibrationModel);
        return std::make_unique<BinaryPredictor<FeatureMatrix, Model>>(
          featureMatrix, model, numLabels, multiThreadingSettings, std::move(binaryTransformationPtr));
    }

    /**
     * Allows to create instances of the type `IBinaryPredictor` that allow to predict known label vectors for given
     * query examples by comparing the predicted scores or probability estimates to the label vectors encountered in the
     * training data.
     */
    class ExampleWiseBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            const std::unique_ptr<IDistanceMeasureFactory<float64>> distanceMeasureFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param distanceMeasureFactoryPtr             An unique pointer to an object of type
             *                                              `IDistanceMeasureFactory` that allows to create
             *                                              implementations of the distance measure that should be used
             *                                              to calculate the distance between predicted scores and known
             *                                              label vectors
             * @param marginalProbabilityCalibrationModel   A pointer to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` to be used for the
             *                                              calibration of marginal probabilities or a null pointer, if
             *                                              no such model is available
             * @param jointProbabilityCalibrationModel      A pointer to an object of type
             *                                              `IJointProbabilityCalibrationModel` to be used for the
             *                                              calibration of joint probabilities or a null pointer, if no
             *                                              such model is available
             * @param multiThreadingSettings                An object of type `MultiThreadingSettings` that stores the
             *                                              setttings to be used for making predictions for different
             *                                              query examples in parallel
             */
            ExampleWiseBinaryPredictorFactory(
              std::unique_ptr<IDistanceMeasureFactory<float64>> distanceMeasureFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel,
              MultiThreadingSettings multiThreadingSettings)
                : distanceMeasureFactoryPtr_(std::move(distanceMeasureFactoryPtr)),
                  marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel),
                  jointProbabilityCalibrationModel_(jointProbabilityCalibrationModel),
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
                return createPredictor(featureMatrix, model, numLabels, multiThreadingSettings_, labelVectorSet,
                                       *distanceMeasureFactoryPtr_,
                                       marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                                            : marginalProbabilityCalibrationModel,
                                       jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                         : jointProbabilityCalibrationModel);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(
              const CsrView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return createPredictor(featureMatrix, model, numLabels, multiThreadingSettings_, labelVectorSet,
                                       *distanceMeasureFactoryPtr_,
                                       marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                                            : marginalProbabilityCalibrationModel,
                                       jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                         : jointProbabilityCalibrationModel);
            }
    };

    template<typename FeatureMatrix, typename Model>
    static inline std::unique_ptr<ISparseBinaryPredictor> createSparsePredictor(
      const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
      MultiThreadingSettings multiThreadingSettings, const LabelVectorSet* labelVectorSet,
      const IDistanceMeasureFactory<float64>& distanceMeasureFactory,
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) {
        std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
          createBinaryTransformation(labelVectorSet, distanceMeasureFactory, marginalProbabilityCalibrationModel,
                                     jointProbabilityCalibrationModel);
        return std::make_unique<SparseBinaryPredictor<FeatureMatrix, Model>>(
          featureMatrix, model, numLabels, multiThreadingSettings, std::move(binaryTransformationPtr));
    }

    /**
     * Allows to create instances of the type `ISparseBinaryPredictor` that allow to predict known label vectors for
     * given query examples by comparing the predicted scores or probability estimates to the label vectors encountered
     * in the training data.
     */
    class ExampleWiseSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
        private:

            const std::unique_ptr<IDistanceMeasureFactory<float64>> distanceMeasureFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param distanceMeasureFactoryPtr             An unique pointer to an object of type
             *                                              `IDistanceMeasureFactory` that allows to create
             *                                              implementations of the distance measure that should be used
             *                                              to calculate the distance between predicted scores and known
             *                                              label vectors
             * @param marginalProbabilityCalibrationModel   A pointer to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` to be used for the
             *                                              calibration of marginal probabilities or a null pointer, if
             *                                              no such model is available
             * @param jointProbabilityCalibrationModel      A pointer to an object of type
             *                                              `IJointProbabilityCalibrationModel` to be used for the
             *                                              calibration of joint probabilities or a null pointer, if no
             *                                              such model is available
             * @param multiThreadingSettings                An object of type `MultiThreadingSettings` that stores the
             *                                              setttings to be used for making predictions for different
             *                                              query examples in parallel
             */
            ExampleWiseSparseBinaryPredictorFactory(
              std::unique_ptr<IDistanceMeasureFactory<float64>> distanceMeasureFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel,
              MultiThreadingSettings multiThreadingSettings)
                : distanceMeasureFactoryPtr_(std::move(distanceMeasureFactoryPtr)),
                  marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel),
                  jointProbabilityCalibrationModel_(jointProbabilityCalibrationModel),
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
                return createSparsePredictor(
                  featureMatrix, model, numLabels, multiThreadingSettings_, labelVectorSet, *distanceMeasureFactoryPtr_,
                  marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                       : marginalProbabilityCalibrationModel,
                  jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                    : jointProbabilityCalibrationModel);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(
              const CsrView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return createSparsePredictor(
                  featureMatrix, model, numLabels, multiThreadingSettings_, labelVectorSet, *distanceMeasureFactoryPtr_,
                  marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                       : marginalProbabilityCalibrationModel,
                  jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                    : jointProbabilityCalibrationModel);
            }
    };

    static inline std::unique_ptr<IDistanceMeasureFactory<float64>> createDistanceMeasureFactory(
      bool basedOnProbabilities, const IClassificationLossConfig& lossConfig) {
        if (basedOnProbabilities) {
            return lossConfig.createJointProbabilityFunctionFactory();
        } else {
            return lossConfig.create64BitClassificationPreset()->createDistanceMeasureFactory();
        }
    }

    ExampleWiseBinaryPredictorConfig::ExampleWiseBinaryPredictorConfig(
      ReadableProperty<IClassificationLossConfig> lossConfig,
      ReadableProperty<IMultiThreadingConfig> multiThreadingConfig)
        : basedOnProbabilities_(false), lossConfig_(lossConfig), multiThreadingConfig_(multiThreadingConfig) {}

    bool ExampleWiseBinaryPredictorConfig::isBasedOnProbabilities() const {
        return basedOnProbabilities_;
    }

    IExampleWiseBinaryPredictorConfig& ExampleWiseBinaryPredictorConfig::setBasedOnProbabilities(
      bool basedOnProbabilities) {
        basedOnProbabilities_ = basedOnProbabilities;
        return *this;
    }

    bool ExampleWiseBinaryPredictorConfig::isProbabilityCalibrationModelUsed() const {
        return noMarginalProbabilityCalibrationModelPtr_ == nullptr;
    }

    IExampleWiseBinaryPredictorConfig& ExampleWiseBinaryPredictorConfig::setUseProbabilityCalibrationModel(
      bool useProbabilityCalibrationModel) {
        noMarginalProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        noJointProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        return *this;
    }

    std::unique_ptr<IBinaryPredictorFactory> ExampleWiseBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
        std::unique_ptr<IDistanceMeasureFactory<float64>> distanceMeasureFactoryPtr =
          createDistanceMeasureFactory(basedOnProbabilities_, lossConfig_.get());

        if (distanceMeasureFactoryPtr) {
            MultiThreadingSettings multiThreadingSettings =
              multiThreadingConfig_.get().getSettings(featureMatrix, numOutputs);
            return std::make_unique<ExampleWiseBinaryPredictorFactory>(
              std::move(distanceMeasureFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(),
              noJointProbabilityCalibrationModelPtr_.get(), multiThreadingSettings);
        }

        return nullptr;
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> ExampleWiseBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDistanceMeasureFactory<float64>> distanceMeasureFactoryPtr =
          createDistanceMeasureFactory(basedOnProbabilities_, lossConfig_.get());

        if (distanceMeasureFactoryPtr) {
            MultiThreadingSettings multiThreadingSettings =
              multiThreadingConfig_.get().getSettings(featureMatrix, numLabels);
            return std::make_unique<ExampleWiseSparseBinaryPredictorFactory>(
              std::move(distanceMeasureFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(),
              noJointProbabilityCalibrationModelPtr_.get(), multiThreadingSettings);
        }

        return nullptr;
    }

    bool ExampleWiseBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
