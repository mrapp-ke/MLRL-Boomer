/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_sparse.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to create instances of the class `ISparseDecomposableRuleEvaluationFactory` that allow to calculate the
     * predictions of single-output rules, which predict for a single output.
     */
    class DecomposableSingleOutputRuleEvaluationFactory final : public ISparseDecomposableRuleEvaluationFactory {
        private:

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

        public:

            /**
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DecomposableSingleOutputRuleEvaluationFactory(float64 l1RegularizationWeight,
                                                          float64 l2RegularizationWeight);

            std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>> create(
              const DenseDecomposableStatisticVector<float64>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>> create(
              const DenseDecomposableStatisticVector<float64>& statisticVector,
              const PartialIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<uint32>>> create(
              const SparseDecomposableStatisticVector<uint32>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<uint32>>> create(
              const SparseDecomposableStatisticVector<uint32>& statisticVector,
              const PartialIndexVector& indexVector) const override;

            virtual std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float32>>> create(
              const SparseDecomposableStatisticVector<float32>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            virtual std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float32>>> create(
              const SparseDecomposableStatisticVector<float32>& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
