/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/partition_sampling.hpp"
#include "common/stopping/global_pruning.hpp"
#include "boosting/losses/loss.hpp"

namespace boosting {

    /**
     * Allows to configure a method that automatically decides for a method that partitions the available training
     * examples into a training set and a holdout set, depending on whether a holdout set is needed and depending on the
     * loss function.
     */
    class AutomaticPartitionSamplingConfig final : public IPartitionSamplingConfig {
        private:

            const std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr_;

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

        public:

            /**
             * @param globalPruningConfigPtr    A reference to an unique pointer that stores the configuration of the
             *                                  method that is used for pruning entire rules
             * @param lossConfigPtr             A reference to an unique pointer that stores the configuration of the
                                                loss function
             */
            AutomaticPartitionSamplingConfig(const std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr,
                                             const std::unique_ptr<ILossConfig>& lossConfigPtr);

            std::unique_ptr<IPartitionSamplingFactory> createPartitionSamplingFactory() const override;
    };

}
