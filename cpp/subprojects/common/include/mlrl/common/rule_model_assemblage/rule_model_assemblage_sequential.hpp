/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_model_assemblage/default_rule.hpp"
#include "mlrl/common/rule_model_assemblage/rule_model_assemblage.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

/**
 * Allows to configure an algorithm that sequentially induces several rules, optionally starting with a default rule,
 * that are added to a rule-based model.
 */
class SequentialRuleModelAssemblageConfig final : public IRuleModelAssemblageConfig {
    private:

        const ReadableProperty<IDefaultRuleConfig> defaultRuleConfig_;

    public:

        /**
         * @param defaultRuleConfigGetter A `ReadableProperty` that allows to access the `IDefaultRuleConfig` that
         *                                stores the configuration of the default rule
         */
        SequentialRuleModelAssemblageConfig(ReadableProperty<IDefaultRuleConfig> defaultRuleConfigGetter);

        std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory(
          const IRowWiseLabelMatrix& labelMatrix) const override;
};
