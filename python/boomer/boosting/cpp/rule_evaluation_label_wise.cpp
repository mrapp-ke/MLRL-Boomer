#include "rule_evaluation_label_wise.h"
#include "linalg.cpp"

using namespace boosting;


/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied label-wise using L2
 * regularization.
 *
 * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
 */
template<class T>
class RegularizedLabelWiseRuleEvaluation : virtual public ILabelWiseRuleEvaluation {

    private:

        const T& labelIndices_;

        float64 l2RegularizationWeight_;

        LabelWiseEvaluatedPrediction prediction_;

    public:

        /**
         * @param labelIndices              A reference to an object of template type `T` that provides access to
         *                                  the indices of the labels for which the rules may predict
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         */
        RegularizedLabelWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight)
            : labelIndices_(labelIndices), l2RegularizationWeight_(l2RegularizationWeight),
              prediction_(LabelWiseEvaluatedPrediction(labelIndices.getNumElements())) {

        }

        const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(
                const float64* totalSumsOfGradients, float64* sumsOfGradients, const float64* totalSumsOfHessians,
                float64* sumsOfHessians, bool uncovered) override {
            uint32 numPredictions = prediction_.getNumElements();
            LabelWiseEvaluatedPrediction::iterator valueIterator = prediction_.begin();
            LabelWiseEvaluatedPrediction::quality_score_iterator qualityScoreIterator = prediction_.quality_scores_begin();
            float64 overallQualityScore = 0;

            // For each label, calculate a score to be predicted, as well as a corresponding quality score...
            typename T::index_const_iterator indexIterator = labelIndices_.indices_cbegin();

            for (uint32 c = 0; c < numPredictions; c++) {
                float64 sumOfGradients = sumsOfGradients[c];
                float64 sumOfHessians =  sumsOfHessians[c];

                if (uncovered) {
                    uint32 l = indexIterator[c];
                    sumOfGradients = totalSumsOfGradients[l] - sumOfGradients;
                    sumOfHessians = totalSumsOfHessians[l] - sumOfHessians;
                }

                // Calculate the score to be predicted for the current label...
                float64 score = sumOfHessians + l2RegularizationWeight_;
                score = score != 0 ? -sumOfGradients / score : 0;
                valueIterator[c] = score;

                // Calculate the quality score for the current label...
                float64 scorePow = pow(score, 2);
                score = (sumOfGradients * score) + (0.5 * scorePow * sumOfHessians);
                qualityScoreIterator[c] = score + (0.5 * l2RegularizationWeight_ * scorePow);
                overallQualityScore += score;
            }

            // Add the L2 regularization term to the overall quality score...
            overallQualityScore += 0.5 * l2RegularizationWeight_ * l2NormPow(valueIterator, numPredictions);
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
    return std::make_unique<RegularizedLabelWiseRuleEvaluation<FullIndexVector>>(indexVector,
                                                                                 l2RegularizationWeight_);
}

std::unique_ptr<ILabelWiseRuleEvaluation> RegularizedLabelWiseRuleEvaluationFactoryImpl::create(
        const PartialIndexVector& indexVector) const {
    return std::make_unique<RegularizedLabelWiseRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                    l2RegularizationWeight_);
}
