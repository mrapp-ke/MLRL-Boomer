/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_example_wise.hpp"
#include "boosting/binning/label_binning.hpp"
#include "boosting/math/blas.hpp"
#include "boosting/math/lapack.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `BinningExampleWiseRuleEvaluation` that uses equal-width binning.
     */
    class EqualWidthBinningExampleWiseRuleEvaluationFactory final : public IExampleWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

            std::shared_ptr<ILabelBinningFactory> labelBinningFactoryPtr_;

            std::shared_ptr<Blas> blasPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

        public:

            /**
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param labelBinningFactoryPtr    A shared pointer to an object of type `ILabelBinningFactory` that allows
                                                to create the implementation to be used to assign labels to bins
             * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
             *                                  different BLAS routines
             * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
             *                                  different LAPACK routines
             */
            EqualWidthBinningExampleWiseRuleEvaluationFactory(
                float64 l2RegularizationWeight, std::shared_ptr<ILabelBinningFactory> labelBinningFactoryPtr,
                std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr);

            std::unique_ptr<IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector>> createDense(
                const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector>> createDense(
                const PartialIndexVector& indexVector) const override;

    };

}
