/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"
#include "mlrl/common/prediction/probability_calibration_marginal.hpp"
#include "mlrl/common/sampling/partition_sampling.hpp"
#include "mlrl/common/stopping/global_pruning.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a method that automatically decides for a method that partitions the available training
     * examples into a training set and a holdout set, depending on whether a holdout set is needed and depending on the
     * loss function.
     */
    class AutomaticPartitionSamplingConfig final : public IPartitionSamplingConfig {
        private:

            const GetterFunction<IGlobalPruningConfig> globalPruningConfigGetter_;

            const GetterFunction<IMarginalProbabilityCalibratorConfig> marginalProbabilityCalibratorConfigGetter_;

            const GetterFunction<IJointProbabilityCalibratorConfig> jointProbabilityCalibratorConfigGetter_;

        public:

            /**
             * @param globalPruningConfigGetter                 A `GetterFunction` that allows to access the
             *                                                  `IGlobalPruningConfig` that stores the configuration of
             *                                                  the method that is used for pruning entire rules
             * @param marginalProbabilityCalibratorConfigGetter A `GetterFunction` that allows to access the
             *                                                  `IMarginalProbabilityCalibratorConfig` that stores the
             *                                                  configuration of the calibrator that is used to fit a
             *                                                  model for the calibration of marginal probabilities
             * @param jointProbabilityCalibratorConfigGetter    A `GetterFunction` that allows to access the
             *                                                  `IJointProbabilityCalibratorConfig` that stores the
             *                                                  configuration of the calibrator that is used to fit a
             *                                                  model for the calibration of joint probabilities
             */
            AutomaticPartitionSamplingConfig(
              GetterFunction<IGlobalPruningConfig> globalPruningConfigGetter,
              GetterFunction<IMarginalProbabilityCalibratorConfig> marginalProbabilityCalibratorConfigGetter,
              GetterFunction<IJointProbabilityCalibratorConfig> jointProbabilityCalibratorConfigGetter);

            /**
             * @see `IPartitionSamplingConfig::createPartitionSamplingFactory`
             */
            std::unique_ptr<IPartitionSamplingFactory> createPartitionSamplingFactory() const override;
    };

}
