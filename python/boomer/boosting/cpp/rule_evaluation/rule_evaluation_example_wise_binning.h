/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "rule_evaluation_example_wise.h"
#include "../math/blas.h"
#include "../math/lapack.h"


namespace boosting {

    /**
     * Allows to create instances of the class `BinningExampleWiseRuleEvaluation` that uses equal-width binning.
     */
    class EqualWidthBinningExampleWiseRuleEvaluationFactory final : public IExampleWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

            float32 binRatio_;

            std::shared_ptr<Blas> blasPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

        public:

            /**
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binRatio                  A percentage that specifies how many bins should be used to assign
             *                                  labels to
             * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
             *                                  different BLAS routines
             * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
             *                                  different LAPACK routines
             */
            EqualWidthBinningExampleWiseRuleEvaluationFactory(float64 l2RegularizationWeight, float32 binRatio,
                                                              std::shared_ptr<Blas> blasPtr,
                                                              std::shared_ptr<Lapack> lapackPtr);

            std::unique_ptr<IExampleWiseRuleEvaluation> create(const FullIndexVector& indexVector) const override;

            std::unique_ptr<IExampleWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
