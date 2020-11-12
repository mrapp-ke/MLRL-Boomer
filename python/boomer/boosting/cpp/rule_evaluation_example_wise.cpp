#include "rule_evaluation_example_wise.h"
#include "data_example_wise.h"
#include "linalg.h"

using namespace boosting;


/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied example wise using L2 regularization.
 */
class RegularizedExampleWiseRuleEvaluation : public IExampleWiseRuleEvaluation {

    private:

        uint32 numPredictions_;

        float64 l2RegularizationWeight_;

        std::shared_ptr<Blas> blasPtr_;

        std::shared_ptr<Lapack> lapackPtr_;

        EvaluatedPrediction* prediction_;

        LabelWiseEvaluatedPrediction* labelWisePrediction_;

    public:

        /**
         * @param numPredictions            The number of labels for which the rules may predict
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
         *                                  different BLAS routines
         * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
         *                                  different LAPACK routines
         */
        RegularizedExampleWiseRuleEvaluation(uint32 numPredictions, float64 l2RegularizationWeight,
                                             std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
            : numPredictions_(numPredictions), l2RegularizationWeight_(l2RegularizationWeight),
              blasPtr_(std::move(blasPtr)), lapackPtr_(std::move(lapackPtr)), prediction_(nullptr),
              labelWisePrediction_(nullptr) {

        }

        ~RegularizedExampleWiseRuleEvaluation() {
            delete prediction_;
            delete labelWisePrediction_;
        }

        const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(
                const DenseExampleWiseStatisticVector& statisticVector) override {
            if (labelWisePrediction_ == nullptr) {
                labelWisePrediction_ = new LabelWiseEvaluatedPrediction(numPredictions_);
            }

            DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                statisticVector.gradients_cbegin();
            LabelWiseEvaluatedPrediction::score_iterator scoreIterator = labelWisePrediction_->scores_begin();
            LabelWiseEvaluatedPrediction::quality_score_iterator qualityScoreIterator =
                labelWisePrediction_->quality_scores_begin();
            float64 overallQualityScore = 0;

            // For each label, calculate the score to be predicted, as well as a quality score...
            for (uint32 c = 0; c < numPredictions_; c++) {
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
            overallQualityScore += 0.5 * l2RegularizationWeight_ * l2NormPow(scoreIterator, numPredictions_);
            labelWisePrediction_->overallQualityScore = overallQualityScore;
            return *labelWisePrediction_;
        }

        const EvaluatedPrediction& calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector,
                                                                  int dsysvLwork, float64* dsysvTmpArray1,
                                                                  int* dsysvTmpArray2, double* dsysvTmpArray3,
                                                                  float64* dspmvTmpArray) override {
            if (prediction_ == nullptr) {
                prediction_ = new EvaluatedPrediction(numPredictions_);
            }

            EvaluatedPrediction::score_iterator scoreIterator = prediction_->scores_begin();
            DenseExampleWiseStatisticVector::gradient_iterator gradientIterator = statisticVector.gradients_begin();
            DenseExampleWiseStatisticVector::hessian_iterator hessianIterator = statisticVector.hessians_begin();

            // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
            lapackPtr_->dsysv(hessianIterator, gradientIterator, dsysvTmpArray1, dsysvTmpArray2, dsysvTmpArray3,
                              scoreIterator, numPredictions_, dsysvLwork, l2RegularizationWeight_);

            // Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
            float64 overallQualityScore = blasPtr_->ddot(scoreIterator, gradientIterator, numPredictions_);
            blasPtr_->dspmv(hessianIterator, scoreIterator, dspmvTmpArray, numPredictions_);
            overallQualityScore += 0.5 * blasPtr_->ddot(scoreIterator, dspmvTmpArray, numPredictions_);

            // Add the L2 regularization term to the overall quality score...
            overallQualityScore += 0.5 * l2RegularizationWeight_ * l2NormPow(scoreIterator, numPredictions_);
            prediction_->overallQualityScore = overallQualityScore;
            return *prediction_;
        }

};

RegularizedExampleWiseRuleEvaluationFactoryImpl::RegularizedExampleWiseRuleEvaluationFactoryImpl(
        float64 l2RegularizationWeight, std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
    : l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(blasPtr), lapackPtr_(lapackPtr) {

}

std::unique_ptr<IExampleWiseRuleEvaluation> RegularizedExampleWiseRuleEvaluationFactoryImpl::create(
        const FullIndexVector& indexVector) const {
    return std::make_unique<RegularizedExampleWiseRuleEvaluation>(indexVector.getNumElements(), l2RegularizationWeight_,
                                                                  blasPtr_, lapackPtr_);
}

std::unique_ptr<IExampleWiseRuleEvaluation> RegularizedExampleWiseRuleEvaluationFactoryImpl::create(
        const PartialIndexVector& indexVector) const {
    return std::make_unique<RegularizedExampleWiseRuleEvaluation>(indexVector.getNumElements(), l2RegularizationWeight_,
                                                                  blasPtr_, lapackPtr_);
}
