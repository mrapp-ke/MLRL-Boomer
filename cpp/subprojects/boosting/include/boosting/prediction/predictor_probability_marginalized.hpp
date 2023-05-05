/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "boosting/macros.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "common/prediction/predictor_probability.hpp"

namespace boosting {

    class MLRLBOOSTING_API IMarginalizedProbabilityPredictorConfig {
        public:

            virtual ~IMarginalizedProbabilityPredictorConfig() {};

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
             *                                        `IMarginalizedProbabilityPredictorConfig` that allows further
             *                                        configuration of the predictor
             */
            virtual IMarginalizedProbabilityPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) = 0;
    };

    /**
     * Allows to configure a predictor that predicts marginalized probabilities for given query examples, which estimate
     * the chance of individual labels to be relevant, by summing up the scores that are provided by individual rules of
     * an existing rule-based model and comparing the aggregated score vector to the known label vectors according to a
     * certain distance measure. The probability for an individual label calculates as the sum of the distances that
     * have been obtained for all label vectors, where the respective label is specified to be relevant, divided by the
     * total sum of all distances.
     */
    class MarginalizedProbabilityPredictorConfig final : public IMarginalizedProbabilityPredictorConfig,
                                                         public IProbabilityPredictorConfig {
        private:

            bool useProbabilityCalibrationModel_;

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
            MarginalizedProbabilityPredictorConfig(
              const std::unique_ptr<ILossConfig>& lossConfigPtr,
              const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            bool isProbabilityCalibrationModelUsed() const override;

            IMarginalizedProbabilityPredictorConfig& setUseProbabilityCalibrationModel(
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
