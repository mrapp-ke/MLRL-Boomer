#include "rule_evaluation_example_wise.h"
#include "data_example_wise.h"
#include "linalg.h"

using namespace boosting;


/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied example wise using L2 regularization.
 *
 * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
 */
template<class T>
class RegularizedExampleWiseRuleEvaluation : public IExampleWiseRuleEvaluation {

    private:

        const T& labelIndices_;

        float64 l2RegularizationWeight_;

        std::shared_ptr<Blas> blasPtr_;

        std::shared_ptr<Lapack> lapackPtr_;

        EvaluatedPrediction* prediction_;

        LabelWiseEvaluatedPrediction* labelWisePrediction_;

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
            : labelIndices_(labelIndices), l2RegularizationWeight_(l2RegularizationWeight),
              blasPtr_(std::move(blasPtr)), lapackPtr_(std::move(lapackPtr)), prediction_(nullptr),
              labelWisePrediction_(nullptr) {

        }

        ~RegularizedExampleWiseRuleEvaluation() {
            delete prediction_;
            delete labelWisePrediction_;
        }

        const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(const float64* gradients,
                                                                         const float64* hessians) override {
            uint32 numPredictions = labelIndices_.getNumElements();

            if (labelWisePrediction_ == nullptr) {
                labelWisePrediction_ = new LabelWiseEvaluatedPrediction(numPredictions);
            }

            LabelWiseEvaluatedPrediction::score_iterator scoreIterator = labelWisePrediction_->scores_begin();
            LabelWiseEvaluatedPrediction::quality_score_iterator qualityScoreIterator =
                labelWisePrediction_->quality_scores_begin();
            float64 overallQualityScore = 0;

            // For each label, calculate the score to be predicted, as well as a quality score...
            for (uint32 c = 0; c < numPredictions; c++) {
                float64 sumOfGradients = gradients[c];
                uint32 c2 = triangularNumber(c + 1) - 1;
                float64 sumOfHessians = hessians[c2];

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
            labelWisePrediction_->overallQualityScore = overallQualityScore;
            return *labelWisePrediction_;
        }

        const EvaluatedPrediction& calculateExampleWisePrediction(float64* gradients, float64* hessians, int dsysvLwork,
                                                                  float64* dsysvTmpArray1, int* dsysvTmpArray2,
                                                                  double* dsysvTmpArray3,
                                                                  float64* dspmvTmpArray) override {
            uint32 numPredictions = labelIndices_.getNumElements();

            if (prediction_ == nullptr) {
                prediction_ = new EvaluatedPrediction(numPredictions);
            }

            EvaluatedPrediction::score_iterator scoreIterator = prediction_->scores_begin();

            // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
            lapackPtr_->dsysv(hessians, gradients, dsysvTmpArray1, dsysvTmpArray2, dsysvTmpArray3, scoreIterator,
                              numPredictions, dsysvLwork, l2RegularizationWeight_);

            // Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
            float64 overallQualityScore = blasPtr_->ddot(scoreIterator, gradients, numPredictions);
            blasPtr_->dspmv(hessians, scoreIterator, dspmvTmpArray, numPredictions);
            overallQualityScore += 0.5 * blasPtr_->ddot(scoreIterator, dspmvTmpArray, numPredictions);

            // Add the L2 regularization term to the overall quality score...
            overallQualityScore += 0.5 * l2RegularizationWeight_ * l2NormPow(scoreIterator, numPredictions);
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
    return std::make_unique<RegularizedExampleWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                                   blasPtr_, lapackPtr_);
}

std::unique_ptr<IExampleWiseRuleEvaluation> RegularizedExampleWiseRuleEvaluationFactoryImpl::create(
        const PartialIndexVector& indexVector) const {
    return std::make_unique<RegularizedExampleWiseRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                      l2RegularizationWeight_, blasPtr_,
                                                                                      lapackPtr_);
}
