/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/boosting/rule_evaluation/head_type.hpp"
#include "mlrl/boosting/statistics/statistic_format.hpp"
#include "mlrl/common/rule_model_assemblage/default_rule.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a method that automatically decides for a format for storing statistics about the quality of
     * predictions for training examples.
     */
    class AutomaticStatisticsConfig final : public IStatisticsConfig {
        private:

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

            const std::unique_ptr<IHeadConfig>& headConfigPtr_;

            const std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr_;

        public:

            /**
             * @param lossConfigPtr         A reference to an unique pointer that stores the configuration of the loss
             *                              function
             * @param headConfigPtr         A reference to an unique pointer that stores the configuration of the rule
             *                              heads
             * @param defaultRuleConfigPtr  A reference to an unique pointer that stores the configuration of the
             *                              default rule
             */
            AutomaticStatisticsConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                      const std::unique_ptr<IHeadConfig>& headConfigPtr,
                                      const std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr);

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            bool isDense() const override;

            bool isSparse() const override;
    };

};
