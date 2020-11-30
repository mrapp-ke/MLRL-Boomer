/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/rule_evaluation/score_vector_label_wise.h"
#include "../data/vector_dense_label_wise.h"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to calculate the predictions of rule, as well as corresponding
     * quality scores, based on the gradients and Hessians that have been calculated according to a loss function that
     * is applied label-wise.
     */
    class ILabelWiseRuleEvaluation {

        public:

            virtual ~ILabelWiseRuleEvaluation() { };

            /**
             * Calculates the scores to be predicted by a rule, as well as corresponding quality scores, based on the
             * label-wise sums of gradients and Hessians that are covered by the rule.
             *
             * @param statisticVector   A reference to an object of type `DenseLabelWiseStatisticVector` that stores the
             *                          gradients and Hessians
             * @return                  A reference to an object of type `ILabelWiseScoreVector` that stores the
             *                          predicted scores and quality scores
             */
            virtual const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseLabelWiseStatisticVector& statisticVector) = 0;

    };

}
