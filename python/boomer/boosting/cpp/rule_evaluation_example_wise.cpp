#include "rule_evaluation_example_wise.h"
#include "math/math.h"
#include <cstdlib>

using namespace boosting;


/**
 * An abstract base class for all classes that allow to calculate the predictions of rules, as well as corresponding
 * quality scores, based on the gradients and Hessians that have been calculated according to a loss function that is
 * applied example-wise.
 */
class AbstractExampleWiseRuleEvaluation : public IExampleWiseRuleEvaluation {

    private:

        uint32 numPredictions_;

        EvaluatedPrediction* prediction_;

        LabelWiseEvaluatedPrediction* labelWisePrediction_;

        int dsysvLwork_;

        float64* dsysvTmpArray1_;

        int* dsysvTmpArray2_;

        double* dsysvTmpArray3_;

        float64* dspmvTmpArray_;

    protected:

        std::shared_ptr<Lapack> lapackPtr_;

        virtual void calculateLabelWisePrediction(const DenseExampleWiseStatisticVector& statisticVector,
                                                  LabelWiseEvaluatedPrediction& prediction) = 0;

        virtual void calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector,
                                                    EvaluatedPrediction& prediction, int dsysvLwork,
                                                    float64* dsysvTmpArray1, int* dsysvTmpArray2,
                                                    double* dsysvTmpArray3, float64* dspmvTmpArray) = 0;

    public:

        /**
         * @param lapackPtr         A shared pointer to an object of type `Lapack` that allows to execute different
         *                          LAPACK routines
         * @param numPredictions    The number of labels for which the rules may predict
         */
        AbstractExampleWiseRuleEvaluation(std::shared_ptr<Lapack> lapackPtr, uint32 numPredictions)
            : numPredictions_(numPredictions), prediction_(nullptr), labelWisePrediction_(nullptr),
              dsysvTmpArray1_(nullptr), dsysvTmpArray2_(nullptr), dsysvTmpArray3_(nullptr), dspmvTmpArray_(nullptr),
              lapackPtr_(lapackPtr) {

        }

        ~AbstractExampleWiseRuleEvaluation() {
            delete prediction_;
            delete labelWisePrediction_;
            free(dsysvTmpArray1_);
            free(dsysvTmpArray2_);
            free(dsysvTmpArray3_);
            free(dspmvTmpArray_);
        }

        const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(
                const DenseExampleWiseStatisticVector& statisticVector) override {
            if (labelWisePrediction_ == nullptr) {
                labelWisePrediction_ = new LabelWiseEvaluatedPrediction(numPredictions_);
            }

            this->calculateLabelWisePrediction(statisticVector, *labelWisePrediction_);
            return *labelWisePrediction_;
        }

        const EvaluatedPrediction& calculateExampleWisePrediction(
                DenseExampleWiseStatisticVector& statisticVector) override {
            if (prediction_ == nullptr) {
                prediction_ = new EvaluatedPrediction(numPredictions_);
                dsysvTmpArray1_ = (float64*) malloc(numPredictions_ * numPredictions_ * sizeof(float64));
                dsysvTmpArray2_ = (int*) malloc(numPredictions_ * sizeof(int));
                dspmvTmpArray_ = (float64*) malloc(numPredictions_ * sizeof(float64));

                // Query the optimal "lwork" parameter to be used by LAPACK's DSYSV routine...
                dsysvLwork_ = lapackPtr_->queryDsysvLworkParameter(dsysvTmpArray1_, dspmvTmpArray_, numPredictions_);
                dsysvTmpArray3_ = (double*) malloc(dsysvLwork_ * sizeof(double));
            }

            this->calculateExampleWisePrediction(statisticVector, *prediction_, dsysvLwork_, dsysvTmpArray1_,
                                                 dsysvTmpArray2_, dsysvTmpArray3_, dspmvTmpArray_);
            return *prediction_;
        }

};

/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied example wise using L2 regularization.
 */
class RegularizedExampleWiseRuleEvaluation : public AbstractExampleWiseRuleEvaluation {

    private:

        float64 l2RegularizationWeight_;

        std::shared_ptr<Blas> blasPtr_;

    protected:

        void calculateLabelWisePrediction(const DenseExampleWiseStatisticVector& statisticVector,
                                          LabelWiseEvaluatedPrediction& prediction) override {
            DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                statisticVector.gradients_cbegin();
            uint32 numPredictions = prediction.getNumElements();
            LabelWiseEvaluatedPrediction::score_iterator scoreIterator = prediction.scores_begin();
            LabelWiseEvaluatedPrediction::quality_score_iterator qualityScoreIterator =
                prediction.quality_scores_begin();
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
            prediction.overallQualityScore = overallQualityScore;
        }

        void calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector,
                                            EvaluatedPrediction& prediction, int dsysvLwork, float64* dsysvTmpArray1,
                                            int* dsysvTmpArray2, double* dsysvTmpArray3,
                                            float64* dspmvTmpArray) override {
            DenseExampleWiseStatisticVector::gradient_iterator gradientIterator = statisticVector.gradients_begin();
            DenseExampleWiseStatisticVector::hessian_iterator hessianIterator = statisticVector.hessians_begin();
            uint32 numPredictions = prediction.getNumElements();
            EvaluatedPrediction::score_iterator scoreIterator = prediction.scores_begin();

            // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
            lapackPtr_->dsysv(hessianIterator, gradientIterator, dsysvTmpArray1, dsysvTmpArray2, dsysvTmpArray3,
                              scoreIterator, numPredictions, dsysvLwork, l2RegularizationWeight_);

            // Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
            float64 overallQualityScore = blasPtr_->ddot(scoreIterator, gradientIterator, numPredictions);
            blasPtr_->dspmv(hessianIterator, scoreIterator, dspmvTmpArray, numPredictions);
            overallQualityScore += 0.5 * blasPtr_->ddot(scoreIterator, dspmvTmpArray, numPredictions);

            // Add the L2 regularization term to the overall quality score...
            overallQualityScore += 0.5 * l2RegularizationWeight_ * l2NormPow(scoreIterator, numPredictions);
            prediction.overallQualityScore = overallQualityScore;
        }

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
            : AbstractExampleWiseRuleEvaluation(lapackPtr, numPredictions),
              l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(blasPtr) {

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
