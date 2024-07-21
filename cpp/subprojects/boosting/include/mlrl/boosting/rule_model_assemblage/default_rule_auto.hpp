/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/boosting/rule_evaluation/head_type.hpp"
#include "mlrl/boosting/statistics/statistic_format.hpp"
#include "mlrl/common/rule_model_assemblage/default_rule.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a method that automatically decides whether a default rule should be included in a rule-based
     * model or not.
     */
    class AutomaticDefaultRuleConfig final : public IDefaultRuleConfig {
        private:

            const ReadableProperty<IClassificationStatisticsConfig> statisticsConfig_;

            const ReadableProperty<IClassificationLossConfig> lossConfig_;

            const ReadableProperty<IHeadConfig> headConfig_;

        public:

            /**
             * @param statisticsConfigGetter    A `ReadableProperty` that allows to access the
             *                                  `IClassificationStatisticsConfig` that stores the configuration of the
             *                                  statistics
             * @param lossConfigGetter          A `ReadableProperty` that allows to access the
             *                                  `IClassificationLossConfig` that stores the configuration of the loss
             *                                  function
             * @param headConfigGetter          A `ReadableProperty` that allows to access the `IHeadConfig` that stores
             *                                  the configuration of the rule heads
             */
            AutomaticDefaultRuleConfig(ReadableProperty<IClassificationStatisticsConfig> statisticsConfigGetter,
                                       ReadableProperty<IClassificationLossConfig> lossConfigGetter,
                                       ReadableProperty<IHeadConfig> headConfigGetter);

            /**
             * @see `IDefaultRuleConfig::isDefaultRuleUsed`
             */
            bool isDefaultRuleUsed(const IOutputMatrix& outputMatrix) const override;
    };

}
