/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable.hpp"
#include "mlrl/boosting/util/blas.hpp"
#include "mlrl/boosting/util/lapack.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to create instances of the class `INonDecomposableRuleEvaluationFactory` that allow to calculate the
     * predictions of complete rules, which predict for all available outputs.
     */
    class NonDecomposableCompleteRuleEvaluationFactory final : public INonDecomposableRuleEvaluationFactory {
        private:

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    An reference to an object of type `Lapack` that allows to execute BLAS
             *                                  routines
             */
            NonDecomposableCompleteRuleEvaluationFactory(float64 l1RegularizationWeight, float64 l2RegularizationWeight,
                                                         const Blas& blas, const Lapack& lapack);

            std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>> create(
              const DenseNonDecomposableStatisticVector<float64>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>> create(
              const DenseNonDecomposableStatisticVector<float64>& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
