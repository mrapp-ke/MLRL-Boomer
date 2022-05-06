/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/statistics/statistic_format.hpp"
#include "boosting/losses/loss.hpp"
#include "boosting/rule_evaluation/head_type.hpp"
#include "common/rule_induction/rule_model_assemblage.hpp"


namespace boosting {

    /**
     * Allows to configure a method that automatically decides for a format for storing statistics about the labels of
     * the training examples.
     */
    class AutomaticStatisticsConfig : public IStatisticsConfig {

        private:

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

            const std::unique_ptr<IHeadConfig>& headConfigPtr_;

            const std::unique_ptr<IRuleModelAssemblageConfig>& ruleModelAssemblageConfigPtr_;

        public:

            /**
             * @param lossConfigPtr                 A reference to an unique pointer that stores the configuration of
             *                                      the loss function
             * @param headConfigPtr                 A reference to an unique pointer that stores the configuration of
             *                                      the rule heads
             * @param ruleModelAssemblageConfigPtr  A reference to an unique pointer that stores the configuration of
             *                                      the algorithm for the induction of several rules
             */
            AutomaticStatisticsConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                      const std::unique_ptr<IHeadConfig>& headConfigPtr,
                                      const std::unique_ptr<IRuleModelAssemblageConfig>& ruleModelAssemblageConfigPtr);

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
                const Lapack& lapack) const override;

    };

};
