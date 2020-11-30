/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/rule_evaluation/score_vector.h"
#include "../../../common/cpp/rule_evaluation/score_vector_label_wise.h"
#include "../data/vector_dense_example_wise.h"


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

}
