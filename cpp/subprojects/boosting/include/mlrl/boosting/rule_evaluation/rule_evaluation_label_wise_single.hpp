/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/rule_evaluation_label_wise_sparse.hpp"

namespace boosting {

    /**
     * Allows to create instances of the class `ISparseLabelWiseRuleEvaluationFactory` that allow to calculate the
     * predictions of single-output rules, which predict for a single output.
     */
    class LabelWiseSingleOutputRuleEvaluationFactory final : public ISparseLabelWiseRuleEvaluationFactory {
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
            LabelWiseSingleOutputRuleEvaluationFactory(float64 l1RegularizationWeight, float64 l2RegularizationWeight);

            std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector>> create(
              const DenseDecomposableStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector>> create(
              const DenseDecomposableStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector>> create(
              const SparseDecomposableStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector>> create(
              const SparseDecomposableStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
