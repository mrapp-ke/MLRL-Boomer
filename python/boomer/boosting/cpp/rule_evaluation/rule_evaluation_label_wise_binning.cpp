#include "rule_evaluation_label_wise_binning.h"

using namespace boosting;


/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied label-wise using L2 regularization.
 * The labels are assigned to bins based on the corresponding gradients.
 *
 * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
 */
template<class T>
class BinningLabelWiseRuleEvaluation : public ILabelWiseRuleEvaluation {

    private:

        float64 l2RegularizationWeight_;

        uint32 numPositiveBins_;

        uint32 numNegativeBins_;

    public:

        /**
         * @param numPositiveBins           The number of bins to be used for labels that should be predicted
         *                                  positively. Must be at least 1
         * @param numNegativeBins           The number of bins to be used for labels that should be predicted
         *                                  negatively. Must be at least 1
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         */
        BinningLabelWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight, uint32 numPositiveBins,
                                       uint32 numNegativeBins)
            : l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
              numNegativeBins_(numNegativeBins) {

        }

        const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseLabelWiseStatisticVector& statisticVector) override {
            // TODO
        }

};

BinningLabelWiseRuleEvaluationFactory::BinningLabelWiseRuleEvaluationFactory(float64 l2RegularizationWeight,
                                                                             uint32 numPositiveBins,
                                                                             uint32 numNegativeBins)
    : l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
      numNegativeBins_(numNegativeBins) {

}

std::unique_ptr<ILabelWiseRuleEvaluation> BinningLabelWiseRuleEvaluationFactory::create(
        const FullIndexVector& indexVector) const {
    return std::make_unique<BinningLabelWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                             numPositiveBins_, numNegativeBins_);
}

std::unique_ptr<ILabelWiseRuleEvaluation> BinningLabelWiseRuleEvaluationFactory::create(
        const PartialIndexVector& indexVector) const {
    return std::make_unique<BinningLabelWiseRuleEvaluation<PartialIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                                numPositiveBins_, numNegativeBins_);
}
