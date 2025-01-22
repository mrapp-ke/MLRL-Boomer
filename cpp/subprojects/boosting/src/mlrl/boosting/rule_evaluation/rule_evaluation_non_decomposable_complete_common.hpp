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
     * Copies Hessians from an iterator to a matrix of coefficients that may be passed to LAPACK's DSYSV routine.
     *
     * @tparam StatisticType    The type of the Hessians
     * @param hessianIterator   An iterator that provides random access to the Hessians
     * @param coefficients      An iterator, the Hessians should be copied to
     * @param n                 The dimensionality of the matrix of coefficients
     */
    template<typename StatisticType>
    static inline void copyCoefficients(typename View<StatisticType>::const_iterator hessianIterator,
                                        typename View<StatisticType>::iterator coefficients, uint32 n) {
        for (uint32 c = 0; c < n; c++) {
            uint32 offset = c * n;

            for (uint32 r = 0; r <= c; r++) {
                coefficients[offset + r] = *hessianIterator;
                hessianIterator++;
            }
        }
    }

    /**
     * Adds a L2 regularization weight to the diagonal of a matrix of coefficients.
     *
     * @tparam StatisticType            The type of the coefficients
     * @param coefficients              An iterator, the regularization weight should be added to
     * @param numPredictions            The number of coefficients on the diagonal
     * @param l2RegularizationWeight    The L2 regularization weight to be added to the coefficients
     */
    template<typename StatisticType>
    static inline void addL2RegularizationWeight(typename View<StatisticType>::iterator coefficients,
                                                 uint32 numPredictions, float32 l2RegularizationWeight) {
        if (l2RegularizationWeight > 0) {
            for (uint32 i = 0; i < numPredictions; i++) {
                coefficients[(i * numPredictions) + i] += l2RegularizationWeight;
            }
        }
    }

    /**
     * Copies gradients from an iterator to a vector of ordinates that may be passed to LAPACK's DSYSV routine.
     *
     * @tparam StatisticType    The type of the gradients
     * @param gradientIterator  An iterator that provides random access to the gradients
     * @param ordinates         An iterator, the gradients should be copied to
     * @param n                 The number of gradients
     */
    template<typename StatisticType>
    static inline void copyOrdinates(typename View<StatisticType>::const_iterator gradientIterator,
                                     typename View<StatisticType>::iterator ordinates, uint32 n) {
        for (uint32 i = 0; i < n; i++) {
            ordinates[i] = -gradientIterator[i];
        }
    }

    /**
     * Adds a L1 regularization weight to a vector of ordinates.
     *
     * @tparam StatisticType            The type of the ordinates
     * @param ordinates                 An iterator, the L1 regularization weight should be added to
     * @param numPredictions            The number of ordinates
     * @param l1RegularizationWeight    The L1 regularization weight to be added to the ordinates
     **/
    template<typename StatisticType>
    static inline void addL1RegularizationWeight(typename View<StatisticType>::iterator ordinates,
                                                 uint32 numPredictions, float32 l1RegularizationWeight) {
        if (l1RegularizationWeight > 0) {
            for (uint32 i = 0; i < numPredictions; i++) {
                StatisticType gradient = ordinates[i];
                ordinates[i] += getL1RegularizationWeight(gradient, l1RegularizationWeight);
            }
        }
    }

    /**
     * Calculates and returns the overall quality of predictions for several outputs.
     *
     * @tparam StatisticType    The type of predictions, gradients and Hessians
     * @param scores            An iterator that provides random access to the predicted scores
     * @param gradients         An iterator that provides random access to the gradients
     * @param hessians          An iterator that provides random access to the Hessians
     * @param tmpArray          An iterator that should be used by BLAS' DSPMV routine to store temporary values
     * @param numPredictions    The number of predictions
     * @param blas              A reference to an object of type `Blas` that allows to execute different BLAS routines
     * @return                  The quality that has been calculated
     */
    template<typename StatisticType>
    static inline StatisticType calculateOverallQuality(typename View<StatisticType>::iterator scores,
                                                        typename View<StatisticType>::iterator gradients,
                                                        typename View<StatisticType>::iterator hessians,
                                                        typename View<StatisticType>::iterator tmpArray,
                                                        uint32 numPredictions, const Blas& blas) {
        blas.dspmv(hessians, scores, tmpArray, numPredictions);
        return blas.ddot(scores, gradients, numPredictions) + (0.5 * blas.ddot(scores, tmpArray, numPredictions));
    }

    /**
     * Calculates and returns the regularization term.
     *
     * @tparam StatisticType            The type of the predicted scores
     * @param scores                    An iterator that provides random access to the predicted scores
     * @param numPredictions            The number of predictions
     * @param l1RegularizationWeight    The weight of the L1 regularization term
     * @param l2RegularizationWeight    The weight of the L2 regularization term
     */
    template<typename StatisticType>
    static inline StatisticType calculateRegularizationTerm(typename View<StatisticType>::const_iterator scores,
                                                            uint32 numPredictions, float32 l1RegularizationWeight,
                                                            float32 l2RegularizationWeight) {
        StatisticType regularizationTerm;

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

            DenseScoreVector<IndexVector> scoreVector_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param outputIndices             A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the outputs for which the rules may predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             */
            DenseNonDecomposableCompleteRuleEvaluation(const IndexVector& outputIndices, float32 l1RegularizationWeight,
                                                       float32 l2RegularizationWeight, const Blas& blas,
                                                       const Lapack& lapack)
                : AbstractNonDecomposableRuleEvaluation<StatisticVector, IndexVector>(outputIndices.getNumElements(),
                                                                                      lapack),
                  scoreVector_(outputIndices, true), l1RegularizationWeight_(l1RegularizationWeight),
                  l2RegularizationWeight_(l2RegularizationWeight), blas_(blas), lapack_(lapack) {}

            /**
             * @see `IRuleEvaluation::evaluate`
             */
            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numPredictions = scoreVector_.getNumElements();

                // Copy Hessians to the matrix of coefficients and add the L2 regularization weight to its diagonal...
                copyCoefficients<float64>(statisticVector.hessians_cbegin(), this->dsysvTmpArray1_.begin(),
                                          numPredictions);
                addL2RegularizationWeight<float64>(this->dsysvTmpArray1_.begin(), numPredictions,
                                                   l2RegularizationWeight_);

                // Copy gradients to the vector of ordinates and add the L1 regularization weight...
                typename DenseScoreVector<IndexVector>::value_iterator valueIterator = scoreVector_.values_begin();
                copyOrdinates<float64>(statisticVector.gradients_cbegin(), valueIterator, numPredictions);
                addL1RegularizationWeight<float64>(valueIterator, numPredictions, l1RegularizationWeight_);

                // Calculate the scores to be predicted for individual outputs by solving a system of linear
                // equations...
                lapack_.dsysv(this->dsysvTmpArray1_.begin(), this->dsysvTmpArray2_.begin(),
                              this->dsysvTmpArray3_.begin(), valueIterator, numPredictions, this->dsysvLwork_);

                // Calculate the overall quality...
                float64 quality = calculateOverallQuality<float64>(valueIterator, statisticVector.gradients_begin(),
                                                                   statisticVector.hessians_begin(),
                                                                   this->dspmvTmpArray_.begin(), numPredictions, blas_);

                // Evaluate regularization term...
                quality += calculateRegularizationTerm<float64>(valueIterator, numPredictions, l1RegularizationWeight_,
                                                                l2RegularizationWeight_);

                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

}
