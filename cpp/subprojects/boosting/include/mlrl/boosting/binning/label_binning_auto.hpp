/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/binning/label_binning.hpp"
#include "mlrl/boosting/rule_evaluation/regularization.hpp"

namespace boosting {

    /**
     * Allows to configure a method that automatically decides whether label binning should be used or not.
     */
    class AutomaticLabelBinningConfig final : public ILabelBinningConfig {
        private:

            const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr_;

            const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr_;

        public:

            /**
             * @param l1RegularizationConfigPtr A reference to an unique pointer that stores the configuration of the L1
             *                                  regularization
             * @param l2RegularizationConfigPtr A reference to an unique pointer that stores the configuration of the L2
             *                                  regularization
             */
            AutomaticLabelBinningConfig(const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
                                        const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr);

            std::unique_ptr<IDecomposableRuleEvaluationFactory> createDecomposableCompleteRuleEvaluationFactory()
              const override;

            std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
              createDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio, uint32 minOutputs,
                                                                  uint32 maxOutputs) const override;

            std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
              createDecomposableDynamicPartialRuleEvaluationFactory(float32 threshold, float32 exponent) const override;

            std::unique_ptr<INonDecomposableRuleEvaluationFactory> createNonDecomposableCompleteRuleEvaluationFactory(
              const Blas& blas, const Lapack& lapack) const override;

            std::unique_ptr<INonDecomposableRuleEvaluationFactory>
              createNonDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio, uint32 minOutputs,
                                                                     uint32 maxOutputs, const Blas& blas,
                                                                     const Lapack& lapack) const override;

            std::unique_ptr<INonDecomposableRuleEvaluationFactory>
              createNonDecomposableDynamicPartialRuleEvaluationFactory(float32 threshold, float32 exponent,
                                                                       const Blas& blas,
                                                                       const Lapack& lapack) const override;
    };

}
