/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/boosting/util/dll_exports.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a predictor that predicts label-wise probabilities
     * for given query examples by transforming the individual scores that are predicted for each label into
     * probabilities.
     */
    class MLRLBOOSTING_API IOutputWiseProbabilityPredictorConfig {
        public:

            virtual ~IOutputWiseProbabilityPredictorConfig() {}

            /**
             * Returns whether a model for the calibration of probabilities is used, if available, or not.
             *
             * @return True, if a model for the calibration of probabilities is used, if available, false otherwise
             */
            virtual bool isProbabilityCalibrationModelUsed() const = 0;

            /**
             * Sets whether a model for the calibration of probabilities should be used, if available, or not.
             *
             * @param useProbabilityCalibrationModel    True, if a model for the calibration of probabilities should be
             *                                          used, if available, false otherwise
             * @return                                  A reference to an object of type
             *                                          `IOutputWiseProbabilityPredictorConfig` that allows further
             *                                          configuration of the predictor
             */
            virtual IOutputWiseProbabilityPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) = 0;
    };

    /**
     * Allows to configure a predictor that predicts label-wise probabilities for given query examples by transforming
     * the individual scores that are predicted for each label into probabilities.
     */
    class OutputWiseProbabilityPredictorConfig final : public IOutputWiseProbabilityPredictorConfig,
                                                       public IProbabilityPredictorConfig {
        private:

            std::unique_ptr<IMarginalProbabilityCalibrationModel> noMarginalProbabilityCalibrationModelPtr_;

            const ReadableProperty<IClassificationLossConfig> lossConfig_;

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

        public:

            /**
             * @param lossConfigGetter            A `ReadableProperty` that allows to access the
             *                                    `IClassificationLossConfig` that stores the configuration of the loss
             *                                    function
             * @param multiThreadingConfigGetter  A `ReadableProperty` that allows to access the `IMultiThreadingConfig`
             *                                    that stores the configuration of the multi-threading behavior that
             *                                    should be used to predict for several query examples in parallel
             */
            OutputWiseProbabilityPredictorConfig(ReadableProperty<IClassificationLossConfig> lossConfigGetter,
                                                 ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter);

            bool isProbabilityCalibrationModelUsed() const override;

            IOutputWiseProbabilityPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) override;

            /**
             * @see `IProbabilityPredictorConfig::createPredictorFactory`
             */
            std::unique_ptr<IProbabilityPredictorFactory> createPredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
