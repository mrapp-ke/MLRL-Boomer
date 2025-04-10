/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/rule_refinement/rule_refinement.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

/**
 * Allows to configure a method for finding the best refinements of existing rules based on statistics for individual
 * examples.
 */
class StatisticsBasedRuleRefinementConfig final : public IRuleRefinementConfig {
    private:

        const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

    public:

        /**
         * @param multiThreadingConfig A `ReadableProperty` that allows to access the `IMultiThreadingConfig` that
         *                             stores the configuration of the multi-threading behavior that should be used for
         *                             the parallel refinement of rules
         */
        StatisticsBasedRuleRefinementConfig(ReadableProperty<IMultiThreadingConfig> multiThreadingConfig);

        std::unique_ptr<IRuleRefinementFactory> createRuleRefinementFactory(const IFeatureMatrix& featureMatrix,
                                                                            uint32 numOutputs) const override;
};
