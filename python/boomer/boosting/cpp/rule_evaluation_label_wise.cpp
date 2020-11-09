#include "rule_evaluation_label_wise.h"
#include "linalg.h"

using namespace boosting;


/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied label-wise using L2 regularization.
 */
class RegularizedLabelWiseRuleEvaluation : public ILabelWiseRuleEvaluation {

    private:

        float64 l2RegularizationWeight_;

        LabelWiseEvaluatedPrediction prediction_;

    public:

        /**
         * @param numPredictions            The number of labels for which the rules may predict
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         */
        RegularizedLabelWiseRuleEvaluation(uint32 numPredictions, float64 l2RegularizationWeight)
            : l2RegularizationWeight_(l2RegularizationWeight),
              prediction_(LabelWiseEvaluatedPrediction(numPredictions)) {

        }

        const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(
                const float64* sumsOfGradients, const float64* sumsOfHessians) override {
            uint32 numPredictions = prediction_.getNumElements();
            LabelWiseEvaluatedPrediction::score_iterator scoreIterator = prediction_.scores_begin();
            LabelWiseEvaluatedPrediction::quality_score_iterator qualityScoreIterator =
                prediction_.quality_scores_begin();
            float64 overallQualityScore = 0;

            // For each label, calculate a score to be predicted, as well as a corresponding quality score...
            for (uint32 c = 0; c < numPredictions; c++) {
                float64 sumOfGradients = sumsOfGradients[c];
                float64 sumOfHessians =  sumsOfHessians[c];

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
            prediction_.overallQualityScore = overallQualityScore;
            return prediction_;
        }

};

RegularizedLabelWiseRuleEvaluationFactoryImpl::RegularizedLabelWiseRuleEvaluationFactoryImpl(
        float64 l2RegularizationWeight)
    : l2RegularizationWeight_(l2RegularizationWeight) {

}

std::unique_ptr<ILabelWiseRuleEvaluation> RegularizedLabelWiseRuleEvaluationFactoryImpl::create(
        const FullIndexVector& indexVector) const {
    return std::make_unique<RegularizedLabelWiseRuleEvaluation>(indexVector.getNumElements(), l2RegularizationWeight_);
}

std::unique_ptr<ILabelWiseRuleEvaluation> RegularizedLabelWiseRuleEvaluationFactoryImpl::create(
        const PartialIndexVector& indexVector) const {
    return std::make_unique<RegularizedLabelWiseRuleEvaluation>(indexVector.getNumElements(), l2RegularizationWeight_);
}
