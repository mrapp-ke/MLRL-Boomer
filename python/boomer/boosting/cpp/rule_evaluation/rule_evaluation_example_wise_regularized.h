/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "rule_evaluation_factory_example_wise.h"
#include "../math/blas.h"
#include "../math/lapack.h"


namespace boosting {

    /**
     * Allows to create instances of the class `RegularizedExampleWiseRuleEvaluation`.
     */
    class RegularizedExampleWiseRuleEvaluationFactoryImpl : public IExampleWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

            std::shared_ptr<Blas> blasPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

        public:

            /**
             * @param l2RegularizationWeight The weight of the L2 regularization that is applied for calculating the
             *                               scores to be predicted by rules
             * @param blasPtr                A shared pointer to an object of type `Blas` that allows to execute
             *                               different BLAS routines
             * @param lapackPtr              A shared pointer to an object of type `Lapack` that allows to execute
             *                               different LAPACK routines
             */
            RegularizedExampleWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight,
                                                            std::shared_ptr<Blas> blasPtr,
                                                            std::shared_ptr<Lapack> lapackPtr);

            std::unique_ptr<IExampleWiseRuleEvaluation> create(const FullIndexVector& indexVector) const override;

            std::unique_ptr<IExampleWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
