/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/boosting/util/dll_exports.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a predictor that predicts known label vectors for
     * given query examples by comparing the predicted scores or probability estimates to the label vectors encountered
     * in the training data.
     */
    class MLRLBOOSTING_API IExampleWiseBinaryPredictorConfig {
        public:

            virtual ~IExampleWiseBinaryPredictorConfig() {}

            /**
             * Returns whether binary predictions are derived from probability estimates rather than scores or not.
             *
             * @return True, if binary predictions are derived from probability estimates rather than scores, false
             *         otherwise
             */
            virtual bool isBasedOnProbabilities() const = 0;

            /**
             * Sets whether binary predictions should be derived from probability estimates rather than scores or not.
             *
             * @param basedOnProbabilities  True, if binary predictions should be derived from probability estimates
             *                              rather than scores, false otherwise
             * @return                      A reference to an object of type `IExampleWiseBinaryPredictorConfig` that
             *                              allows further configuration of the predictor
             */
            virtual IExampleWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities) = 0;

            /**
             * Returns whether a model for the calibration of probabilities is used, if available, or not.
             *
             * @return True, if a model for the calibration of probabilities is used, if available, false otherwise
             */
            virtual bool isProbabilityCalibrationModelUsed() const = 0;

            /**
             * Sets whether a model for the calibration of probabilities should be used, if available, or not.
             *
             * @param useProbabilityCalibrationModel  True, if a model for the calibration of probabilities should be
             *                                        used, if available, false otherwise
             * @return                                A reference to an object of type
             *                                        `IExampleWiseBinaryPredictorConfig` that allows further
             *                                        configuration of the predictor
             */
            virtual IExampleWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) = 0;
    };

    /**
     * Allows to configure a predictor that predicts known label vectors for given query examples by comparing the
     * predicted scores or probability estimates to the label vectors encountered in the training data.
     */
    class ExampleWiseBinaryPredictorConfig final : public IExampleWiseBinaryPredictorConfig,
                                                   public IBinaryPredictorConfig {
        private:

            bool basedOnProbabilities_;

            std::unique_ptr<IMarginalProbabilityCalibrationModel> noMarginalProbabilityCalibrationModelPtr_;

            std::unique_ptr<IJointProbabilityCalibrationModel> noJointProbabilityCalibrationModelPtr_;

            const ReadableProperty<IClassificationLossConfig> lossConfig_;

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

        public:

            /**
             * @param lossConfig            A `ReadableProperty` that allows to access the `IClassificationLossConfig`
             *                              that stores the configuration of the loss function
             * @param multiThreadingConfig  A `ReadableProperty` that allows to access the `IMultiThreadingConfig` that
             *                              stores the configuration of the multi-threading behavior that should be used
             *                              to predict for several query examples in parallel
             */
            ExampleWiseBinaryPredictorConfig(ReadableProperty<IClassificationLossConfig> lossConfig,
                                             ReadableProperty<IMultiThreadingConfig> multiThreadingConfig);

            bool isBasedOnProbabilities() const override;

            IExampleWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities) override;

            bool isProbabilityCalibrationModelUsed() const override;

            IExampleWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) override;

            /**
             * @see `IPredictorConfig::createPredictorFactory`
             */
            std::unique_ptr<IBinaryPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                            uint32 numOutputs) const override;

            /**
             * @see `IBinaryPredictorConfig::createSparsePredictorFactory`
             */
            std::unique_ptr<ISparseBinaryPredictorFactory> createSparsePredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
