#include "rule_evaluation_example_wise_regularized.h"
#include "rule_evaluation_example_wise_common.h"
#include "../../../common/cpp/rule_evaluation/score_vector_label_wise_dense.h"
#include "../math/math.h"

using namespace boosting;


/**
 * Copies the Hessians that are stored by a vector to a coefficient matrix that may be passed to LAPACK's DSYSV routine.
 *
 * @tparam StatisticVector  The type of the vector that stores the Hessians
 * @param statisticVector   A reference to an object of template type `StatisticVector` that stores the Hessians
 * @param output            A pointer to an array of type `float64`, shape `(n, n)`, the Hessians should be copied to
 * @param n                 The number of rows and columns in the coefficient matrix
 */
template<class StatisticVector>
static inline void copyCoefficients(const StatisticVector& statisticVector, float64* output, uint32 n) {
    typename StatisticVector::hessian_const_iterator hessianIterator = statisticVector.hessians_cbegin();

    for (uint32 c = 0; c < n; c++) {
        uint32 offset = c * n;

        for (uint32 r = 0; r < c + 1; r++) {
            float64 hessian = *hessianIterator;
            output[offset + r] = hessian;
            hessianIterator++;
        }
    }
}

/**
 * Adds a specific L2 regularization weight to the diagonal of a coefficient matrix.
 *
 * @param output                    A pointer to an array of type `float64`, shape `(n, n)` that stores the coefficients
 * @param n                         The number of rows and columns in the coefficient matrix
 * @param l2RegularizationWeight    The L2 regularization weight to be added
 */
static inline void addRegularizationWeight(float64* output, uint32 n, float64 l2RegularizationWeight) {
    for (uint32 i = 0; i < n; i++) {
        output[(i * n) + i] += l2RegularizationWeight;
    }
}

/**
 * Copies the gradients that are stored by a vector to a vector of ordinates that may be passed to LAPACK's DSYSV
 * routine.
 *
 * @tparam StatisticVector  The type of the vector that stores the gradients
 * @param statisticVector   A reference to an object of template type `StatisticVector` that stores the gradients
 * @param output            A pointer to an array of type `float64`, shape `(n)`, the gradients should be copied to
 * @param n                 The number of ordinates
 */
template<class StatisticVector>
static inline void copyOrdinates(const StatisticVector& statisticVector, float64* output, uint32 n) {
    typename StatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();

    for (uint32 i = 0; i < n; i++) {
        float64 gradient = *gradientIterator;
        output[i] = -gradient;
        gradientIterator++;
    }
}

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

        DenseScoreVector<T>* scoreVector_;

        DenseLabelWiseScoreVector<T>* labelWiseScoreVector_;

        void calculateLabelWisePrediction(const DenseExampleWiseStatisticVector& statisticVector,
                                          DenseLabelWiseScoreVector<T>& scoreVector) {
            DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                statisticVector.gradients_cbegin();
            DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator =
                statisticVector.hessians_diagonal_cbegin();
            uint32 numPredictions = scoreVector.getNumElements();
            typename DenseLabelWiseScoreVector<T>::score_iterator scoreIterator = scoreVector.scores_begin();
            typename DenseLabelWiseScoreVector<T>::quality_score_iterator qualityScoreIterator =
                scoreVector.quality_scores_begin();
            float64 overallQualityScore = 0;

            // For each label, calculate the score to be predicted, as well as a quality score...
            for (uint32 c = 0; c < numPredictions; c++) {
                float64 sumOfGradients = gradientIterator[c];
                float64 sumOfHessians = hessianIterator[c];

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
                                            float64* dspmvTmpArray) {
            uint32 numPredictions = scoreVector.getNumElements();
            typename DenseScoreVector<T>::score_iterator scoreIterator = scoreVector.scores_begin();

            // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
            copyCoefficients(statisticVector, dsysvTmpArray1, numPredictions);
            addRegularizationWeight(dsysvTmpArray1, numPredictions, l2RegularizationWeight_);
            copyOrdinates(statisticVector, scoreIterator, numPredictions);
            this->lapackPtr_->dsysv(dsysvTmpArray1, dsysvTmpArray2, dsysvTmpArray3, scoreIterator, numPredictions,
                                    dsysvLwork);

            // Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
            float64 overallQualityScore = blasPtr_->ddot(scoreIterator, statisticVector.gradients_begin(),
                                                         numPredictions);
            blasPtr_->dspmv(statisticVector.hessians_begin(), scoreIterator, dspmvTmpArray, numPredictions);
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
              l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(blasPtr), scoreVector_(nullptr),
              labelWiseScoreVector_(nullptr) {

        }

        const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseExampleWiseStatisticVector& statisticVector) override {
            if (labelWiseScoreVector_ == nullptr) {
                labelWiseScoreVector_ = new DenseLabelWiseScoreVector<T>(this->labelIndices_);
            }

            this->calculateLabelWisePrediction(statisticVector, *labelWiseScoreVector_);
            return *labelWiseScoreVector_;
        }

        const IScoreVector& calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector) override {
            uint32 numPredictions = this->labelIndices_.getNumElements();

            if (scoreVector_ == nullptr) {
                scoreVector_ = new DenseScoreVector<T>(this->labelIndices_);
                this->initializeTmpArrays(numPredictions);
            }

            this->calculateExampleWisePrediction(statisticVector, *scoreVector_, this->dsysvLwork_,
                                                 this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_,
                                                 this->dspmvTmpArray_);
            return *scoreVector_;
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
