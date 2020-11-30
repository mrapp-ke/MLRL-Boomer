#include "rule_evaluation_label_wise.h"
#include "math/math.h"
#include "../../common/cpp/rule_evaluation/score_vector_label_wise_dense.h"

using namespace boosting;


/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied label-wise using L2 regularization.
 *
 * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
 */
template<class T>
class RegularizedLabelWiseRuleEvaluation : public ILabelWiseRuleEvaluation {

    private:

        float64 l2RegularizationWeight_;

        DenseLabelWiseScoreVector<T> scoreVector_;

    public:

        /**
         * @param labelIndices              A reference to an object of template type `T` that provides access to
         *                                  the indices of the labels for which the rules may predict
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         */
        RegularizedLabelWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight)
            : l2RegularizationWeight_(l2RegularizationWeight),
              scoreVector_(DenseLabelWiseScoreVector<T>(labelIndices)) {

        }

        const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseLabelWiseStatisticVector& statisticVector) override {
            DenseLabelWiseStatisticVector::gradient_const_iterator gradientIterator =
                statisticVector.gradients_cbegin();
            DenseLabelWiseStatisticVector::hessian_const_iterator hessianIterator =
                statisticVector.hessians_cbegin();
            uint32 numPredictions = scoreVector_.getNumElements();
            typename DenseLabelWiseScoreVector<T>::score_iterator scoreIterator = scoreVector_.scores_begin();
            typename DenseLabelWiseScoreVector<T>::quality_score_iterator qualityScoreIterator =
                scoreVector_.quality_scores_begin();
            float64 overallQualityScore = 0;

            // For each label, calculate a score to be predicted, as well as a corresponding quality score...
            for (uint32 c = 0; c < numPredictions; c++) {
                float64 sumOfGradients = gradientIterator[c];
                float64 sumOfHessians =  hessianIterator[c];

                // Calculate the score to be predicted for the current label...
                float64 score = sumOfHessians + l2RegularizationWeight_;
                score = score != 0 ? -sumOfGradients / score : 0;
                scoreIterator[c] = score;

                // Calculate the quality score for the current label...
                float64 scorePow = score * score;
                score = (sumOfGradients * score) + (0.5 * scorePow * sumOfHessians);
                qualityScoreIterator[c] = score + (0.5 * l2RegularizationWeight_ * scorePow);
                overallQualityScore += score;
            }

            // Add the L2 regularization term to the overall quality score...
            overallQualityScore += 0.5 * l2RegularizationWeight_ * l2NormPow(scoreIterator, numPredictions);
            scoreVector_.overallQualityScore = overallQualityScore;
            return scoreVector_;
        }

};

RegularizedLabelWiseRuleEvaluationFactoryImpl::RegularizedLabelWiseRuleEvaluationFactoryImpl(
        float64 l2RegularizationWeight)
    : l2RegularizationWeight_(l2RegularizationWeight) {

}

std::unique_ptr<ILabelWiseRuleEvaluation> RegularizedLabelWiseRuleEvaluationFactoryImpl::create(
        const FullIndexVector& indexVector) const {
    return std::make_unique<RegularizedLabelWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_);
}

std::unique_ptr<ILabelWiseRuleEvaluation> RegularizedLabelWiseRuleEvaluationFactoryImpl::create(
        const PartialIndexVector& indexVector) const {
    return std::make_unique<RegularizedLabelWiseRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                    l2RegularizationWeight_);
}
