#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_dynamic.hpp"
#include "rule_evaluation_example_wise_complete_common.hpp"
#include "rule_evaluation_example_wise_partial_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a subset of the available labels that is
     * determined dynamically, as well as an overall quality score, based on the gradients and Hessians that are stored
     * by a `DenseExampleWiseStatisticVector` using L1 and L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseExampleWiseDynamicPartialRuleEvaluation final :
            public AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, T> {

        private:

            const T& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            float64 threshold_;

            float64 l1RegularizationWeight_;

            float64 l2RegularizationWeight_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             */
            DenseExampleWiseDynamicPartialRuleEvaluation(const T& labelIndices, float32 threshold,
                                                         float64 l1RegularizationWeight, float64 l2RegularizationWeight,
                                                         const Blas& blas, const Lapack& lapack)
                : AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, T>(labelIndices.getNumElements(),
                                                                                        lapack),
                  labelIndices_(labelIndices), indexVector_(PartialIndexVector(labelIndices.getNumElements())),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_, true)), threshold_(1.0 - threshold),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
                  blas_(blas), lapack_(lapack) {

            }

            const IScoreVector& calculatePrediction(DenseExampleWiseStatisticVector& statisticVector) override {
                uint32 numLabels = statisticVector.getNumElements();
                DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                    statisticVector.gradients_cbegin();
                DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator =
                    statisticVector.hessians_diagonal_cbegin();
                typename DenseScoreVector<T>::score_iterator scoreIterator = scoreVector_.scores_begin();
                float64 bestScore = calculateLabelWiseScore(gradientIterator[0], hessianIterator[0],
                                                            l1RegularizationWeight_, l2RegularizationWeight_);
                scoreIterator[0] = bestScore;

                for (uint32 i = 1; i < numLabels; i++) {
                    float64 score = calculateLabelWiseScore(gradientIterator[i], hessianIterator[i],
                                                            l1RegularizationWeight_, l2RegularizationWeight_);
                    scoreIterator[i] = score;

                    if (std::abs(score) > std::abs(bestScore)) {
                        bestScore = score;
                    }
                }

                // Copy gradients to the vector of ordinates and add the L1 regularization weight...
                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                typename T::const_iterator labelIndexIterator = labelIndices_.cbegin();
                float64 threshold = (bestScore * bestScore) * threshold_;
                uint32 n = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 score = scoreIterator[i];

                    if (score * score > threshold) {
                        indexIterator[n] = labelIndexIterator[i];
                        scoreIterator[n] = -gradientIterator[i];
                        n++;
                    }
                }

                indexVector_.setNumElements(n, false);
                addL1RegularizationWeight(scoreIterator, n, l1RegularizationWeight_);

                // Copy Hessians to the matrix of coefficients and add the L2 regularization weight to its diagonal...
                copyCoefficients(statisticVector.hessians_cbegin(), indexIterator, this->dsysvTmpArray1_, n);
                addL2RegularizationWeight(this->dsysvTmpArray1_, n, l2RegularizationWeight_);

                // Calculate the scores to be predicted for individual labels by solving a system of linear equations...
                lapack_.dsysv(this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_, scoreIterator, n,
                              this->dsysvLwork_);

                // Calculate the overall quality score...
                float64 overallQualityScore = calculateOverallQualityScore(scoreIterator,
                                                                           statisticVector.gradients_begin(),
                                                                           statisticVector.hessians_begin(),
                                                                           this->dspmvTmpArray_, n, blas_);

                // Evaluate regularization term...
                overallQualityScore += calculateRegularizationTerm(scoreIterator, n, l1RegularizationWeight_,
                                                                   l2RegularizationWeight_);

                scoreVector_.overallQualityScore = overallQualityScore;
                return scoreVector_;
            }

    };

    ExampleWiseDynamicPartialRuleEvaluationFactory::ExampleWiseDynamicPartialRuleEvaluationFactory(
            float32 threshold, float64 l1RegularizationWeight, float64 l2RegularizationWeight, const Blas& blas,
            const Lapack& lapack)
        : threshold_(threshold), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight), blas_(blas), lapack_(lapack) {

    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseDynamicPartialRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseExampleWiseDynamicPartialRuleEvaluation<CompleteIndexVector>>(
            indexVector, threshold_, l1RegularizationWeight_, l2RegularizationWeight_, blas_, lapack_);
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseDynamicPartialRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseExampleWiseCompleteRuleEvaluation<PartialIndexVector>>(
            indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blas_, lapack_);;
    }

}
