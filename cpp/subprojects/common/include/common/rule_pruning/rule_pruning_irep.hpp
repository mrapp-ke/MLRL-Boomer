/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/pruning/pruning.hpp"


/**
 * Allows to configure a strategy for pruning individual rules that prunes rules by following the principles of
 * "incremental reduced error pruning" (IREP).
 */
class IrepConfig final : public IPruningConfig {

    private:

        RuleCompareFunction ruleCompareFunction_;

    public:

        /**
         * @param compareFunction An object of type `RuleCompareFunction` that defines the function that should be used
         *                        for comparing the quality of different rules
         */
        IrepConfig(RuleCompareFunction compareFunction);

        std::unique_ptr<IPruningFactory> createPruningFactory() const override;

};
