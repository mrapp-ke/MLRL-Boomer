/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/util/math.hpp"
#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "rule_evaluation_decomposable_common.hpp"
#include "rule_evaluation_non_decomposable_common.hpp"

namespace boosting {

    /**
     * Copies Hessians from an iterator to a matrix of coefficients that may be passed to LAPACK's SYSV routine.
     *
     * @tparam HessianIterator      The type of the iterator that provides access to the Hessians
     * @tparam CoefficientIterator  The type of the iterator, the Hessians should be copied to
     * @param hessians              An iterator that provides random access to the Hessians
     * @param coefficients          An iterator, the Hessians should be copied to
     * @param n                     The dimensionality of the matrix of coefficients
     */
    template<typename HessianIterator, typename CoefficientIterator>
    static inline void copyCoefficients(HessianIterator hessians, CoefficientIterator coefficients, uint32 n) {
        for (uint32 c = 0; c < n; c++) {
            uint32 offset = c * n;

            for (uint32 r = 0; r <= c; r++) {
                coefficients[offset + r] = *hessians;
                hessians++;
            }
        }
    }

    /**
     * Adds a L2 regularization weight to the diagonal of a matrix of coefficients.
     *
     * @tparam CoefficientIterator      The type of the iterator that provides access to the coefficients
     * @param coefficients              An iterator, the regularization weight should be added to
     * @param numPredictions            The number of coefficients on the diagonal
     * @param l2RegularizationWeight    The L2 regularization weight to be added to the coefficients
     */
    template<typename CoefficientIterator>
    static inline void addL2RegularizationWeight(CoefficientIterator coefficients, uint32 numPredictions,
                                                 float32 l2RegularizationWeight) {
        if (l2RegularizationWeight > 0) {
            for (uint32 i = 0; i < numPredictions; i++) {
                coefficients[(i * numPredictions) + i] += l2RegularizationWeight;
            }
        }
    }

    /**
     * Copies gradients from an iterator to a vector of ordinates that may be passed to LAPACK's SYSV routine.
     *
     * @tparam GradientIterator The type of the iterator that provides access to the gradients
     * @tparam OrdinateIterator The type of the iterator, the gradients should be copied to
     * @param gradients         An iterator that provides random access to the gradients
     * @param ordinates         An iterator, the gradients should be copied to
     * @param numGradients      The number of gradients
     */
    template<typename GradientIterator, typename OrdinateIterator>
    static inline void copyOrdinates(GradientIterator gradients, OrdinateIterator ordinates, uint32 numGradients) {
        for (uint32 i = 0; i < numGradients; i++) {
            ordinates[i] = -gradients[i];
        }
    }

    /**
     * Adds a L1 regularization weight to a vector of ordinates.
     *
     * @tparam OrdinateIterator         The type of the iterator that provides access to the ordinates
     * @param ordinates                 An iterator, the L1 regularization weight should be added to
     * @param numPredictions            The number of ordinates
     * @param l1RegularizationWeight    The L1 regularization weight to be added to the ordinates
     **/
    template<typename OrdinateIterator>
    static inline void addL1RegularizationWeight(OrdinateIterator ordinates, uint32 numPredictions,
                                                 float32 l1RegularizationWeight) {
        if (l1RegularizationWeight > 0) {
            for (uint32 i = 0; i < numPredictions; i++) {
                typename util::iterator_value<OrdinateIterator> gradient = ordinates[i];
                ordinates[i] += getL1RegularizationWeight(gradient, l1RegularizationWeight);
            }
        }
    }

    /**
     * Calculates and returns the overall quality of predictions for several outputs.
     *
     * @tparam ScoreIterator    The type of the iterator that provides access to the predicted scores
     * @tparam GradientIterator The type of the iterator that provides access to the gradients
     * @tparam HessianIterator  The type of the iterator that provides access to the Hessians
     * @param scores            An iterator that provides random access to the predicted scores
     * @param gradients         An iterator that provides random access to the gradients
     * @param hessians          An iterator that provides random access to the Hessians
     * @param tmpArray          An iterator that should be used by BLAS' SPMV routine to store temporary values
     * @param numPredictions    The number of predictions
     * @param blas              A reference to an object of type `Blas` that allows to execute different BLAS routines
     * @return                  The quality that has been calculated
     */
    template<typename ScoreIterator, typename GradientIterator, typename HessianIterator>
    static inline typename util::iterator_value<GradientIterator> calculateOverallQuality(
      ScoreIterator scores, GradientIterator gradients, HessianIterator hessians,
      typename View<typename util::iterator_value<GradientIterator>>::iterator tmpArray, uint32 numPredictions,
      const Blas<typename util::iterator_value<GradientIterator>>& blas) {
        blas.spmv(hessians, scores, tmpArray, numPredictions);
        return blas.dot(scores, gradients, numPredictions) + (0.5 * blas.dot(scores, tmpArray, numPredictions));
    }

    /**
     * Calculates and returns the regularization term.
     *
     * @tparam ScoreIterator            The type of the iterator that provides access to the predicted scores
     * @param scores                    An iterator that provides random access to the predicted scores
     * @param numPredictions            The number of predictions
     * @param l1RegularizationWeight    The weight of the L1 regularization term
     * @param l2RegularizationWeight    The weight of the L2 regularization term
     */
    template<typename ScoreIterator>
    static inline typename util::iterator_value<ScoreIterator> calculateRegularizationTerm(
      ScoreIterator scores, uint32 numPredictions, float32 l1RegularizationWeight, float32 l2RegularizationWeight) {
        typename util::iterator_value<ScoreIterator> regularizationTerm;

        if (l1RegularizationWeight > 0) {
            regularizationTerm = l1RegularizationWeight * util::l1Norm(scores, numPredictions);
        } else {
            regularizationTerm = 0;
        }

        if (l2RegularizationWeight > 0) {
            regularizationTerm += 0.5 * l2RegularizationWeight * util::l2NormPow(scores, numPredictions);
        }

        return regularizationTerm;
    }

    /**
     * Allows to calculate the predictions of complete rules, as well as their overall quality, based on the gradients
     * and Hessians that are stored by a `DenseNonDecomposableStatisticVector` using L1 and L2 regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the outputs for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DenseNonDecomposableCompleteRuleEvaluation final
        : public AbstractNonDecomposableRuleEvaluation<StatisticVector, IndexVector> {
        private:

            typedef typename StatisticVector::statistic_type statistic_type;

            DenseScoreVector<statistic_type, IndexVector> scoreVector_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

            const std::unique_ptr<Blas<statistic_type>> blasPtr_;

            const std::unique_ptr<Lapack<statistic_type>> lapackPtr_;

        public:

            /**
             * @param outputIndices             A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the outputs for which the rules may predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blasPtr                   An unique pointer to an object of type `Blas` that allows to execute
             *                                  BLAS routines
             * @param lapackPtr                 An unique pointer to an object of type `Lapack` that allows to execute
             *                                  LAPACK routines
             */
            DenseNonDecomposableCompleteRuleEvaluation(const IndexVector& outputIndices, float32 l1RegularizationWeight,
                                                       float32 l2RegularizationWeight,
                                                       std::unique_ptr<Blas<statistic_type>> blasPtr,
                                                       std::unique_ptr<Lapack<statistic_type>> lapackPtr)
                : AbstractNonDecomposableRuleEvaluation<StatisticVector, IndexVector>(outputIndices.getNumElements(),
                                                                                      *lapackPtr),
                  scoreVector_(outputIndices, true), l1RegularizationWeight_(l1RegularizationWeight),
                  l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(std::move(blasPtr)),
                  lapackPtr_(std::move(lapackPtr)) {}

            /**
             * @see `IRuleEvaluation::evaluate`
             */
            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numPredictions = scoreVector_.getNumElements();

                // Copy Hessians to the matrix of coefficients and add the L2 regularization weight to its diagonal...
                copyCoefficients(statisticVector.hessians_cbegin(), this->sysvTmpArray1_.begin(), numPredictions);
                addL2RegularizationWeight(this->sysvTmpArray1_.begin(), numPredictions, l2RegularizationWeight_);

                // Copy gradients to the vector of ordinates and add the L1 regularization weight...
                typename DenseScoreVector<statistic_type, IndexVector>::value_iterator valueIterator =
                  scoreVector_.values_begin();
                copyOrdinates(statisticVector.gradients_cbegin(), valueIterator, numPredictions);
                addL1RegularizationWeight(valueIterator, numPredictions, l1RegularizationWeight_);

                // Calculate the scores to be predicted for individual outputs by solving a system of linear
                // equations...
                lapackPtr_->sysv(this->sysvTmpArray1_.begin(), this->sysvTmpArray2_.begin(),
                                 this->sysvTmpArray3_.begin(), valueIterator, numPredictions, this->sysvLwork_);

                // Calculate the overall quality...
                statistic_type quality = calculateOverallQuality(
                  valueIterator, statisticVector.gradients_begin(), statisticVector.hessians_begin(),
                  this->spmvTmpArray_.begin(), numPredictions, *blasPtr_);

                // Evaluate regularization term...
                quality += calculateRegularizationTerm(valueIterator, numPredictions, l1RegularizationWeight_,
                                                       l2RegularizationWeight_);

                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

}
