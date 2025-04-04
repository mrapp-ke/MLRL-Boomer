/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/binning/label_binning.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable.hpp"
#include "mlrl/boosting/util/blas.hpp"
#include "mlrl/boosting/util/lapack.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to create instances of the class `INonDecomposableRuleEvaluationFactory` that allow to calculate the
     * predictions of partial rules, which predict for a subset of the available outputs that is determined dynamically,
     * using gradient-based label binning.
     */
    class NonDecomposableDynamicPartialBinnedRuleEvaluationFactory final
        : public INonDecomposableRuleEvaluationFactory {
        private:

            const float32 threshold_;

            const float32 exponent_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

            const std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr_;

            const BlasFactory& blasFactory_;

            const LapackFactory& lapackFactory_;

        public:

            /**
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict. A smaller threshold results in less labels being selected. A
             *                                  greater threshold results in more labels being selected. E.g., a
             *                                  threshold of 0.2 means that a rule will only predict for a label if the
             *                                  estimated predictive quality `q` for this particular label satisfies the
             *                                  inequality `q^exponent > q_best^exponent * (1 - 0.2)`, where `q_best` is
             *                                  the best quality among all labels. Must be in (0, 1)
             * @param exponent                  An exponent that should be used to weigh the estimated predictive
             *                                  quality for individual labels. E.g., an exponent of 2 means that the
             *                                  estimated predictive quality `q` for a particular label is weighed as
             *                                  `q^2`. Must be at least 1
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param labelBinningFactoryPtr    An unique pointer to an object of type `ILabelBinningFactory` that
             *                                  allows to create the implementation to be used to assign labels to bins
             * @param blasFactory               A reference to an object of type `BlasFactory` that allows to create
             *                                  objects for executing BLAS routines
             * @param lapackFactory             An reference to an object of type `Lapack` that allows to create objects
             *                                  for executing BLAS routines
             */
            NonDecomposableDynamicPartialBinnedRuleEvaluationFactory(
              float32 threshold, float32 exponent, float32 l1RegularizationWeight, float32 l2RegularizationWeight,
              std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr, const BlasFactory& blasFactory,
              const LapackFactory& lapackFactory);

            std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float32>>> create(
              const DenseNonDecomposableStatisticVector<float32>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float32>>> create(
              const DenseNonDecomposableStatisticVector<float32>& statisticVector,
              const PartialIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>> create(
              const DenseNonDecomposableStatisticVector<float64>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>> create(
              const DenseNonDecomposableStatisticVector<float64>& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
