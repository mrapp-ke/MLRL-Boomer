/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/pruning/pruning.hpp"


/**
 * Allows to configure a strategy for pruning classification rules that prunes rules by following the ideas of
 * "incremental reduced error pruning" (IREP).
 */
class IrepConfig final : public IPruningConfig {

    private:

        Quality::CompareFunction ruleCompareFunction_;

    public:

        /**
         * @param ruleCompareFunction The function that should be used for comparing the quality of different rules
         */
        IrepConfig(Quality::CompareFunction compareFunction);

        std::unique_ptr<IPruningFactory> createPruningFactory() const override;

};
