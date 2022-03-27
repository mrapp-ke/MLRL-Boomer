/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_example_wise.hpp"
#include "boosting/binning/label_binning.hpp"
#include "boosting/math/blas.hpp"
#include "boosting/math/lapack.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `IExampleWiseRuleEvaluationFactory` that allow to calculate the
     * predictions of partial rules, which predict for a subset of the available labels that is determined dynamically,
     * using gradient-based label binning.
     */
    class ExampleWiseDynamicPartialBinnedRuleEvaluationFactory final : public IExampleWiseRuleEvaluationFactory {

        private:

            float32 threshold_;

            float64 l1RegularizationWeight_;

            float64 l2RegularizationWeight_;

            std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict. A smaller threshold results in less labels being selected. A
             *                                  greater threshold results in more labels being selected. E.g., a
             *                                  threshold of 0.2 means that a rule will only predict for a label if the
             *                                  estimated predictive quality `q` for this particular label satisfies the
             *                                  inequality `q^2 > q_best^2 * (1 - 0.2)`, where `q_best` is the best
             *                                  quality among all labels. Must be in (0, 1)
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param labelBinningFactoryPtr    An unique pointer to an object of type `ILabelBinningFactory` that
             *                                  allows to create the implementation to be used to assign labels to bins
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    An reference to an object of type `Lapack` that allows to execute BLAS
             *                                  routines
             */
            ExampleWiseDynamicPartialBinnedRuleEvaluationFactory(
                float32 threshold, float64 l1RegularizationWeight, float64 l2RegularizationWeight,
                std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr, const Blas& blas, const Lapack& lapack);

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
                const DenseExampleWiseStatisticVector& statisticVector,
                const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
                const DenseExampleWiseStatisticVector& statisticVector,
                const PartialIndexVector& indexVector) const override;

    };

}
