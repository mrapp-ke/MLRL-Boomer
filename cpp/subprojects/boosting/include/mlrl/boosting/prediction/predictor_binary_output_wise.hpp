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
     * Defines an interface for all classes that allow to configure a predictor that predicts whether individual labels
     * of given query examples are relevant or irrelevant by discretizing the individual scores or probability estimates
     * that are predicted for each label.
     */
    class MLRLBOOSTING_API IOutputWiseBinaryPredictorConfig {
        public:

            virtual ~IOutputWiseBinaryPredictorConfig() {}

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
             * @return                      A reference to an object of type `IOutputWiseBinaryPredictorConfig` that
             *                              allows further configuration of the predictor
             */
            virtual IOutputWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities) = 0;

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
             *                                        `IOutputWiseBinaryPredictorConfig` that allows further
             *                                        configuration of the predictor
             */
            virtual IOutputWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) = 0;
    };

    /**
     * Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
     * irrelevant by discretizing the individual scores or probability estimates that are predicted for each label.
     */
    class OutputWiseBinaryPredictorConfig final : public IOutputWiseBinaryPredictorConfig,
                                                  public IBinaryPredictorConfig {
        private:

            bool basedOnProbabilities_;

            std::unique_ptr<IMarginalProbabilityCalibrationModel> noMarginalProbabilityCalibrationModelPtr_;

            const ReadableProperty<ILossConfig> lossConfig_;

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

        public:

            /**
             * @param lossConfigGetter            A `ReadableProperty` that allows to access the `ILossConfig` that
             *                                    stores the configuration of the loss function
             * @param multiThreadingConfigGetter  A `ReadableProperty` that allows to access the `IMultiThreadingConfig`
             *                                    that stores the configuration of the multi-threading behavior that
             *                                    should be used to predict for several query examples in parallel
             */
            OutputWiseBinaryPredictorConfig(ReadableProperty<ILossConfig> lossConfigGetter,
                                            ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter);

            bool isBasedOnProbabilities() const override;

            IOutputWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities) override;

            bool isProbabilityCalibrationModelUsed() const override;

            IOutputWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) override;

            /**
             * @see `IPredictorFactory::createPredictorFactory`
             */
            std::unique_ptr<IBinaryPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                            uint32 numOutputs) const override;

            /**
             * @see `IBinaryPredictorFactory::createSparsePredictorFactory`
             */
            std::unique_ptr<ISparseBinaryPredictorFactory> createSparsePredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
