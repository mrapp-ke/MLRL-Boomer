/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/binning/label_binning.hpp"
#include "mlrl/boosting/rule_evaluation/regularization.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

#include <memory>

namespace boosting {

    /**
     * Allows to configure a method that does not assign labels to bins.
     */
    class NoLabelBinningConfig final : public ILabelBinningConfig {
        private:

            const ReadableProperty<IRegularizationConfig> l1RegularizationConfig_;

            const ReadableProperty<IRegularizationConfig> l2RegularizationConfig_;

        public:

            /**
             * @param l1RegularizationConfig  A `ReadableProperty` that allows to access the `IRegularizationConfig`
             *                                that stores the configuration of the L1 regularization
             * @param l2RegularizationConfig  A `ReadableProperty` that allows to access the `IRegularizationConfig`
             *                                that stores the configuration of the L2 regularization
             */
            NoLabelBinningConfig(ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                                 ReadableProperty<IRegularizationConfig> l2RegularizationConfig);

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
