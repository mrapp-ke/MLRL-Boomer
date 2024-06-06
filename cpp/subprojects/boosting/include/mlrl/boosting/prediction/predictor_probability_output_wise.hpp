/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/boosting/util/dll_exports.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"

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

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

        public:

            /**
             * @param lossConfigPtr             A reference to an unique pointer that stores the configuration of the
             *                                  loss function
             * @param multiThreadingConfigPtr   A reference to an unique pointer that stores the configuration of the
             *                                  multi-threading behavior that should be used to predict for several
             *                                  query examples in parallel
             */
            OutputWiseProbabilityPredictorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                                 const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            bool isProbabilityCalibrationModelUsed() const override;

            IOutputWiseProbabilityPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) override;

            /**
             * @see `IProbabilityPredictorConfig::createPredictorFactory`
             */
            std::unique_ptr<IProbabilityPredictorFactory> createPredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
