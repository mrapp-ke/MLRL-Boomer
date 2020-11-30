#include "rule_evaluation_example_wise_regularized.h"
#include "rule_evaluation_example_wise_common.h"
#include "../math/math.h"

using namespace boosting;


/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied example wise using L2 regularization.
 *
 * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
 */
template<class T>
class RegularizedExampleWiseRuleEvaluation : public AbstractExampleWiseRuleEvaluation<T> {

    private:

        float64 l2RegularizationWeight_;

        std::shared_ptr<Blas> blasPtr_;

    protected:

        void calculateLabelWisePrediction(const DenseExampleWiseStatisticVector& statisticVector,
                                          DenseLabelWiseScoreVector<T>& scoreVector) override {
            DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                statisticVector.gradients_cbegin();
            uint32 numPredictions = scoreVector.getNumElements();
            typename DenseLabelWiseScoreVector<T>::score_iterator scoreIterator = scoreVector.scores_begin();
            typename DenseLabelWiseScoreVector<T>::quality_score_iterator qualityScoreIterator =
                scoreVector.quality_scores_begin();
            float64 overallQualityScore = 0;

            // For each label, calculate the score to be predicted, as well as a quality score...
            for (uint32 c = 0; c < numPredictions; c++) {
                float64 sumOfGradients = gradientIterator[c];
                float64 sumOfHessians = statisticVector.hessian_diagonal(c);

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
            scoreVector.overallQualityScore = overallQualityScore;
        }

        void calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector,
                                            DenseScoreVector<T>& scoreVector, int dsysvLwork, float64* dsysvTmpArray1,
                                            int* dsysvTmpArray2, double* dsysvTmpArray3,
                                            float64* dspmvTmpArray) override {
            DenseExampleWiseStatisticVector::gradient_iterator gradientIterator = statisticVector.gradients_begin();
            DenseExampleWiseStatisticVector::hessian_iterator hessianIterator = statisticVector.hessians_begin();
            uint32 numPredictions = scoreVector.getNumElements();
            typename DenseScoreVector<T>::score_iterator scoreIterator = scoreVector.scores_begin();

            // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
            this->lapackPtr_->dsysv(hessianIterator, gradientIterator, dsysvTmpArray1, dsysvTmpArray2, dsysvTmpArray3,
                                    scoreIterator, numPredictions, dsysvLwork, l2RegularizationWeight_);

            // Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
            float64 overallQualityScore = blasPtr_->ddot(scoreIterator, gradientIterator, numPredictions);
            blasPtr_->dspmv(hessianIterator, scoreIterator, dspmvTmpArray, numPredictions);
            overallQualityScore += 0.5 * blasPtr_->ddot(scoreIterator, dspmvTmpArray, numPredictions);

            // Add the L2 regularization term to the overall quality score...
            overallQualityScore += 0.5 * l2RegularizationWeight_ * l2NormPow(scoreIterator, numPredictions);
            scoreVector.overallQualityScore = overallQualityScore;
        }

    public:

        /**
         * @param labelIndices              A reference to an object of template type `T` that provides access to the
         *                                  indices of the labels for which the rules may predict
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
         *                                  different BLAS routines
         * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
         *                                  different LAPACK routines
         */
        RegularizedExampleWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight,
                                             std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
            : AbstractExampleWiseRuleEvaluation<T>(labelIndices, lapackPtr),
              l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(blasPtr) {

        }

};

RegularizedExampleWiseRuleEvaluationFactory::RegularizedExampleWiseRuleEvaluationFactory(
        float64 l2RegularizationWeight, std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
    : l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(blasPtr), lapackPtr_(lapackPtr) {

}

std::unique_ptr<IExampleWiseRuleEvaluation> RegularizedExampleWiseRuleEvaluationFactory::create(
        const FullIndexVector& indexVector) const {
    return std::make_unique<RegularizedExampleWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                                   blasPtr_, lapackPtr_);
}

std::unique_ptr<IExampleWiseRuleEvaluation> RegularizedExampleWiseRuleEvaluationFactory::create(
        const PartialIndexVector& indexVector) const {
    return std::make_unique<RegularizedExampleWiseRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                      l2RegularizationWeight_, blasPtr_,
                                                                                      lapackPtr_);
}
