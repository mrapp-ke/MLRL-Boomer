/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "rule_evaluation_example_wise.h"
#include "../math/blas.h"
#include "../math/lapack.h"


namespace boosting {

    /**
     * Allows to create instances of the class `BinningExampleWiseRuleEvaluation`.
     */
    class BinningExampleWiseRuleEvaluationFactory : public IExampleWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

            uint32 numPositiveBins_;

            uint32 numNegativeBins_;

            std::shared_ptr<Blas> blasPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

        public:

            /**
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param numPositiveBins           The number of bins to be used for labels that should be predicted
             *                                  positively
             * @param numNegativeBins           The number of bins to be used for labels that should be predicted
             *                                  negatively
             * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
             *                                  different BLAS routines
             * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
             *                                  different LAPACK routines
             */
            BinningExampleWiseRuleEvaluationFactory(float64 l2RegularizationWeight, uint32 numPositiveBins,
                                                    uint32 numNegativeBins, std::shared_ptr<Blas> blasPtr,
                                                    std::shared_ptr<Lapack> lapackPtr);

            std::unique_ptr<IExampleWiseRuleEvaluation> create(const FullIndexVector& indexVector) const override;

            std::unique_ptr<IExampleWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
