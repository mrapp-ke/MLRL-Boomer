/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_induction/rule_induction.hpp"
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

        const ReadableProperty<IRuleInductionConfig> ruleInductionConfig_;

        const ReadableProperty<IDefaultRuleConfig> defaultRuleConfig_;

    public:

        /**
         * @param ruleInductionConfig   A `ReadableProperty` that allows to access the `IRuleInductionConfig` that
         *                              stores the configuration of the algorithm for the induction of individual rules
         * @param defaultRuleConfig     A `ReadableProperty` that allows to access the `IDefaultRuleConfig` that stores
         *                              the configuration of the default rule
         */
        SequentialRuleModelAssemblageConfig(ReadableProperty<IRuleInductionConfig> ruleInductionConfig,
                                            ReadableProperty<IDefaultRuleConfig> defaultRuleConfig);

        std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory(
          const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const override;
};
