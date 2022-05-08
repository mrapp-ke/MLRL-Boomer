/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_induction/default_rule.hpp"
#include "boosting/rule_evaluation/head_type.hpp"


namespace boosting {

    /**
     * Allows to configure a method that automatically decides whether a default rule should be included in a rule-based
     * model or not.
     */
    class AutomaticDefaultRuleConfig final : public IDefaultRuleConfig {

        private:

            const std::unique_ptr<IHeadConfig>& headConfigPtr_;

        public:

            /**
             * @param headConfigPtr A reference to an unique pointer that stores the configuration of the rule heads
             */
            AutomaticDefaultRuleConfig(const std::unique_ptr<IHeadConfig>& headConfigPtr);

            bool isDefaultRuleUsed(const IRowWiseLabelMatrix& labelMatrix) const override;

    };

}
