/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_induction/rule_model_assemblage.hpp"
#include "common/rule_induction/default_rule.hpp"


/**
 * Allows to configure an algorithm that sequentially induces several rules, optionally starting with a default rule,
 * that are added to a rule-based model.
 */
class SequentialRuleModelAssemblageConfig final : public IRuleModelAssemblageConfig {

    private:

        const std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr_;

    public:

        /**
         * @param defaultRuleConfigPtr A reference to an unique pointer that stores the configuration of the default
         *                             rule
         */
        SequentialRuleModelAssemblageConfig(const std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr);

        std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory(
            const IRowWiseLabelMatrix& labelMatrix) const override;

};
