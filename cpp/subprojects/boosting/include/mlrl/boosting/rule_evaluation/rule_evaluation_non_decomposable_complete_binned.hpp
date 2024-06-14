/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/binning/label_binning.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable.hpp"
#include "mlrl/boosting/util/blas.hpp"
#include "mlrl/boosting/util/lapack.hpp"

namespace boosting {

    /**
     * Allows to create instances of the class `INonDecomposableRuleEvaluationFactory` that allow to calculate the
     * predictions of complete rules, which predict for all available labels, using gradient-based label binning.
     */
    class NonDecomposableCompleteBinnedRuleEvaluationFactory final : public INonDecomposableRuleEvaluationFactory {
        private:

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            const std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param labelBinningFactoryPtr    An unique pointer to an object of type `ILabelBinningFactory` that
             *                                  allows to create the implementation to be used to assign labels to bins
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             */
            NonDecomposableCompleteBinnedRuleEvaluationFactory(
              float64 l1RegularizationWeight, float64 l2RegularizationWeight,
              std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr, const Blas& blas, const Lapack& lapack);

            std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector>> create(
              const DenseNonDecomposableStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector>> create(
              const DenseNonDecomposableStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
