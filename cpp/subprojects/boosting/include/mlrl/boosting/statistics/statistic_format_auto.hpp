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
     * Allows to configure a method that automatically decides for a format for storing statistics about the quality of
     * predictions for training examples.
     */
    class AutomaticStatisticsConfig final : public IStatisticsConfig {
        private:

            const GetterFunction<ILossConfig> lossConfigGetter_;

            const GetterFunction<IHeadConfig> headConfigGetter_;

            const GetterFunction<IDefaultRuleConfig> defaultRuleConfigGetter_;

        public:

            /**
             * @param lossConfigGetter          A `GetterFunction` that allows to access the `ILossConfig` that stores
             *                                  the configuration of the loss function
             * @param headConfigGetter          A `GetterFunction` that allows to access the `IHeadConfig` that stores
             *                                  the configuration of the rule heads
             * @param defaultRuleConfigGetter   A `GetterFunction` that allows to access the `IDefaultRuleConfig` that
             *                                  stores the configuration of the default rule
             */
            AutomaticStatisticsConfig(GetterFunction<ILossConfig> lossConfigGetter,
                                      GetterFunction<IHeadConfig> headConfigGetter,
                                      GetterFunction<IDefaultRuleConfig> defaultRuleConfigGetter);

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            bool isDense() const override;

            bool isSparse() const override;
    };

};
