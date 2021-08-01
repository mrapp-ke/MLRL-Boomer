#include "boosting/rule_evaluation/rule_evaluation_example_wise_regularized.hpp"
#include "boosting/math/math.hpp"
#include "common/rule_evaluation/score_vector_label_wise_dense.hpp"
#include "common/validation.hpp"
#include "rule_evaluation_label_wise_regularized_common.hpp"
#include "rule_evaluation_example_wise_common.hpp"


namespace boosting {

    /**
     * Adds a specific L2 regularization weight to the diagonal of a coefficient matrix.
     *
     * @param output                    A pointer to an array of type `float64`, shape `(n, n)` that stores the
     *                                  coefficients
     * @param n                         The number of rows and columns in the coefficient matrix
     * @param l2RegularizationWeight    The L2 regularization weight to be added
     */
    static inline void addRegularizationWeight(float64* output, uint32 n, float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < n; i++) {
            output[(i * n) + i] += l2RegularizationWeight;
        }
    }

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
     * Hessians that are stored by a `DenseExampleWiseStatisticVector` using L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseRegularizedExampleWiseRuleEvaluation final :
            public AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, T> {

        private:

            float64 l2RegularizationWeight_;

            const Blas& blas_;

            DenseScoreVector<T>* scoreVector_;

            DenseLabelWiseScoreVector<T>* labelWiseScoreVector_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blas                      A reference to an object of type `Blas` that allows to execute different
             *                                  BLAS routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute
             *                                  different LAPACK routines
             */
            DenseRegularizedExampleWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight,
                                                      const Blas& blas, const Lapack& lapack)
                : AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, T>(labelIndices, lapack),
                  l2RegularizationWeight_(l2RegularizationWeight), blas_(blas), scoreVector_(nullptr),
                  labelWiseScoreVector_(nullptr) {

            }

            /**
             * @see `IExampleWiseRuleEvaluation::calculateLabelWisePrediction`
             */
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

            /**
             * @see `IExampleWiseRuleEvaluation::calculateExampleWisePrediction`
             */
            const IScoreVector& calculateExampleWisePrediction(
                    DenseExampleWiseStatisticVector& statisticVector) override {
                uint32 numPredictions = this->labelIndices_.getNumElements();

                if (scoreVector_ == nullptr) {
                    scoreVector_ = new DenseScoreVector<T>(this->labelIndices_);
                    this->initializeTmpArrays(numPredictions);
                }

                typename DenseScoreVector<T>::score_iterator scoreIterator = scoreVector_->scores_begin();
                copyCoefficients<DenseExampleWiseStatisticVector::hessian_const_iterator>(
                    statisticVector.hessians_cbegin(), this->dsysvTmpArray1_, numPredictions);
                addRegularizationWeight(this->dsysvTmpArray1_, numPredictions, l2RegularizationWeight_);
                copyOrdinates<DenseExampleWiseStatisticVector::gradient_const_iterator>(
                    statisticVector.gradients_cbegin(), scoreIterator, numPredictions);

                // Calculate the scores to be predicted for the individual labels by solving a system of linear
                // equations...
                this->lapack_.dsysv(this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_, scoreIterator,
                                    numPredictions, this->dsysvLwork_);

                // Calculate the overall quality score...
                float64 qualityScore = calculateExampleWiseQualityScore(numPredictions, scoreIterator,
                                                                        statisticVector.gradients_begin(),
                                                                        statisticVector.hessians_begin(), blas_,
                                                                        this->dspmvTmpArray_);
                qualityScore += 0.5 * l2RegularizationWeight_ * l2NormPow<typename DenseScoreVector<T>::score_iterator>(
                    scoreIterator, numPredictions);
                scoreVector_->overallQualityScore = qualityScore;
                return *scoreVector_;
            }

            /**
             * @see `IExampleWiseRuleEvaluation::calculatePrediction`
             */
            const IScoreVector& calculatePrediction(const DenseExampleWiseStatisticVector& statisticVector) override {
                // TODO Implement
                return *scoreVector_;
            }

    };

    RegularizedExampleWiseRuleEvaluationFactory::RegularizedExampleWiseRuleEvaluationFactory(
            float64 l2RegularizationWeight, std::unique_ptr<Blas> blasPtr, std::unique_ptr<Lapack> lapackPtr)
        : l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(std::move(blasPtr)),
          lapackPtr_(std::move(lapackPtr)) {
        assertNotNull("blasPtr", blasPtr_.get());
        assertNotNull("lapackPtr", lapackPtr_.get());
        assertGreaterOrEqual<float64>("l2RegularizationWeight", l2RegularizationWeight, 0);
    }

    std::unique_ptr<IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector>> RegularizedExampleWiseRuleEvaluationFactory::createDense(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseRegularizedExampleWiseRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                                l2RegularizationWeight_,
                                                                                                *blasPtr_, *lapackPtr_);
    }

    std::unique_ptr<IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector>> RegularizedExampleWiseRuleEvaluationFactory::createDense(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseRegularizedExampleWiseRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                               l2RegularizationWeight_,
                                                                                               *blasPtr_, *lapackPtr_);

    }

}
