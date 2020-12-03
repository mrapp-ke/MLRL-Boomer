/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "rule_evaluation_label_wise.h"


namespace boosting {

    /**
     * Allows to create instances of the class `BinningLabelWiseRuleEvaluation` that uses equal-width binning.
     */
    class EqualWidthBinningLabelWiseRuleEvaluationFactory : public ILabelWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

            uint32 numPositiveBins_;

            uint32 numNegativeBins_;

        public:

            /**
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param numPositiveBins           The number of bins to be used for labels that should be predicted
             *                                  positively
             * @param numNegativeBins           The number of bins to be used for label that should be predicted
             *                                  negatively
             */
            EqualWidthBinningLabelWiseRuleEvaluationFactory(float64 l2RegularizationWeight, uint32 numPositiveBins,
                                                            uint32 numNegativeBins);

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const FullIndexVector& indexVector) const override;

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
