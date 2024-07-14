/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/stopping/global_pruning.hpp"

#include <memory>

/**
 * Allows to configure a stopping criterion that does not actually perform any global pruning.
 */
class NoGlobalPruningConfig final : public IGlobalPruningConfig {
    public:

        std::unique_ptr<IStoppingCriterionFactory> createStoppingCriterionFactory() const override;

        /**
         * @see `IGlobalPruningConfig::shouldUseHoldoutSet`
         */
        bool shouldUseHoldoutSet() const override;

        /**
         * @see `IGlobalPruningConfig::shouldRemoveUnusedRules`
         */
        bool shouldRemoveUnusedRules() const override;
};
