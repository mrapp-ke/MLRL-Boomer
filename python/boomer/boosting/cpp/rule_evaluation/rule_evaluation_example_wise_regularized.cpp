#include "rule_evaluation_example_wise_regularized.h"
#include "rule_evaluation_example_wise_common.h"
#include "rule_evaluation_label_wise_regularized_common.h"
#include "../../../common/cpp/rule_evaluation/score_vector_label_wise_dense.h"
#include "../math/math.h"

using namespace boosting;


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
class RegularizedExampleWiseRuleEvaluation final : public AbstractExampleWiseRuleEvaluation<T> {

    private:

        float64 l2RegularizationWeight_;

        std::shared_ptr<Blas> blasPtr_;

        DenseScoreVector<T>* scoreVector_;

        DenseLabelWiseScoreVector<T>* labelWiseScoreVector_;

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

            labelWiseScoreVector_->overallQualityScore = calculateLabelWisePredictionInternally<
                    typename DenseLabelWiseScoreVector<T>::score_iterator,
                    typename DenseLabelWiseScoreVector<T>::quality_score_iterator,
                    DenseExampleWiseStatisticVector::gradient_const_iterator,
                    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator>(
                labelWiseScoreVector_->getNumElements(), labelWiseScoreVector_->scores_begin(),
                labelWiseScoreVector_->quality_scores_begin(), statisticVector.gradients_cbegin(),
                statisticVector.hessians_diagonal_cbegin(), l2RegularizationWeight_);
            return *labelWiseScoreVector_;
        }

        const IScoreVector& calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector) override {
            uint32 numPredictions = this->labelIndices_.getNumElements();

            if (scoreVector_ == nullptr) {
                scoreVector_ = new DenseScoreVector<T>(this->labelIndices_);
                this->initializeTmpArrays(numPredictions);
            }

            typename DenseScoreVector<T>::score_iterator scoreIterator = scoreVector_->scores_begin();
            copyCoefficients<DenseExampleWiseStatisticVector::hessian_const_iterator>(
                statisticVector.hessians_cbegin(), this->dsysvTmpArray1_, numPredictions);
            addRegularizationWeight(this->dsysvTmpArray1_, numPredictions, l2RegularizationWeight_);
            copyOrdinates<DenseExampleWiseStatisticVector>(statisticVector, scoreIterator, numPredictions);
            scoreVector_->overallQualityScore = calculateExampleWisePredictionInternally(
                numPredictions, scoreIterator, statisticVector.gradients_begin(), statisticVector.hessians_begin(),
                l2RegularizationWeight_, *blasPtr_, *this->lapackPtr_, this->dsysvLwork_, this->dsysvTmpArray1_,
                this->dsysvTmpArray2_, this->dsysvTmpArray3_, this->dspmvTmpArray_);
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
