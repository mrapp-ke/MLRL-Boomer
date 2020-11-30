/**
 * Implements classes for calculating the predictions of rules, as well as corresponding quality scores, based on the
 * gradients and Hessians that have been calculated according to a loss function that is applied example-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/rule_evaluation/score_vector.h"
#include "../../common/cpp/rule_evaluation/score_vector_label_wise.h"
#include "data/vector_dense_example_wise.h"
#include "math/blas.h"
#include "math/lapack.h"
#include <memory>


namespace boosting {

    /**
     * Defines an interface for all classes that allow to calculate the predictions of rule, as well as corresponding
     * quality scores, based on the gradients and Hessians that have been calculated according to a loss function that
     * is applied example-wise.
     */
    class IExampleWiseRuleEvaluation {

        public:

            virtual ~IExampleWiseRuleEvaluation() { };

            /**
             * Calculates the scores to be predicted by a rule, as well as corresponding quality scores, based on the
             * label-wise sums of gradients and Hessians that are covered by the rule.
             *
             * @param statisticVector   A reference to an object of type `DenseExampleWiseStatisticVector` that stores
             *                          the gradients and Hessians
             * @param return            A reference to an object of type `ILabelWiseScoreVector` that stores the
             *                          predicted scores and quality scores
             */
            virtual const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseExampleWiseStatisticVector& statisticVector) = 0;

            /**
             * Calculates the scores to be predicted by a rule, as well as an overall quality score, based on the sums
             * of gradients and Hessians that are covered by the rule.
             *
             * @param statisticVector   A reference to an object of type `DenseExampleWiseStatisticVector` that stores
             *                          the gradients and Hessians
             * @param prediction        A reference to an object of type `IScoreVector` that should be used to store the
             *                          predicted scores and quality score
             */
            virtual const IScoreVector& calculateExampleWisePrediction(
                DenseExampleWiseStatisticVector& statisticVector) = 0;

    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `IExampleWiseRuleEvaluation`.
     */
    class IExampleWiseRuleEvaluationFactory {

        public:

            virtual ~IExampleWiseRuleEvaluationFactory() { };

            /**
             * Creates and returns a new object of type `ILabelWiseRuleEvaluation` that allows to calculate the
             * predictions of rules that predict for all available labels.
             *
             * @param indexVector   A reference to an object of type `FullIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<IExampleWiseRuleEvaluation> create(const FullIndexVector& indexVector) const = 0;

            /**
             * Creates and returns a new object of type `ILabelWiseRuleEvaluation` that allows to calculate the
             * predictions of rules that predict for a subset of the available labels.
             *
             * @param indexVector   A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<IExampleWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const = 0;

    };

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
