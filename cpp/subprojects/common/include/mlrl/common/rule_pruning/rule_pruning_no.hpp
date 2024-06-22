/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_pruning/rule_pruning.hpp"

#include <memory>

/**
 * Allows to configure a method for pruning individual rules that does not actually perform any pruning.
 */
class NoRulePruningConfig final : public IRulePruningConfig {
    public:

        std::unique_ptr<IRulePruningFactory> createRulePruningFactory() const override;
};
