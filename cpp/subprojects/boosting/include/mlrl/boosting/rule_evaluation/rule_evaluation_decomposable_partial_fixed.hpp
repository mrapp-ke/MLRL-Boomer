/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_sparse.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to create instances of the class `ISparseDecomposableRuleEvaluationFactory` that allow to calculate the
     * predictions of partial rules, which predict for a predefined number of outputs.
     */
    class DecomposableFixedPartialRuleEvaluationFactory final : public ISparseDecomposableRuleEvaluationFactory {
        private:

            const float32 outputRatio_;

            const uint32 minOutputs_;

            const uint32 maxOutputs_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

        public:

            /**
             * @param outputRatio               A percentage that specifies for how many outputs the rule heads should
             *                                  predict, e.g., if 100 outputs are available, a percentage of 0.5 means
             *                                  that the rule heads predict for a subset of `ceil(0.5 * 100) = 50`
             *                                  outputs. Must be in (0, 1)
             * @param minOutputs                The minimum number of outputs for which the rule heads should predict.
             *                                  Must be at least 2
             * @param maxOutputs                The maximum number of outputs for which the rule heads should predict.
             *                                  Must be at least `minOutputs` or 0, if the maximum number of outputs
             *                                  should not be restricted
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio, uint32 minOutputs, uint32 maxOutputs,
                                                          float32 l1RegularizationWeight,
                                                          float32 l2RegularizationWeight);

            std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>> create(
              const DenseDecomposableStatisticVector<float64>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>> create(
              const DenseDecomposableStatisticVector<float64>& statisticVector,
              const PartialIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>>> create(
              const SparseDecomposableStatisticVector<float64, uint32>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>>> create(
              const SparseDecomposableStatisticVector<float64, uint32>& statisticVector,
              const PartialIndexVector& indexVector) const override;

            virtual std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>>> create(
              const SparseDecomposableStatisticVector<float64, float32>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            virtual std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>>> create(
              const SparseDecomposableStatisticVector<float64, float32>& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
