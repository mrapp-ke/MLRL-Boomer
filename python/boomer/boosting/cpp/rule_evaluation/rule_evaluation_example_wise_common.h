/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "rule_evaluation_example_wise.h"
#include "../math/blas.h"
#include "../math/lapack.h"
#include "../math/math.h"
#include <cstdlib>


namespace boosting {

    /**
     * Copies the Hessians that are stored by a vector to a coefficient matrix that may be passed to LAPACK's DSYSV
     * routine.
     *
     * @tparam HessianIterator  The type of the iterator that provides access to the Hessians
     * @param hessianIterator   An iterator of template type `HessianIterator` that provides random access to the
     *                          Hessians that are stored in the vector
     * @param output            A pointer to an array of type `float64`, shape `(n, n)`, the Hessians should be copied
     *                          to
     * @param n                 The number of rows and columns in the coefficient matrix
     */
    template<class HessianIterator>
    static inline void copyCoefficients(HessianIterator hessianIterator, float64* output, uint32 n) {
        uint32 i = 0;

        for (uint32 c = 0; c < n; c++) {
            uint32 offset = c * n;

            for (uint32 r = 0; r < c + 1; r++) {
                float64 hessian = hessianIterator[i];
                output[offset + r] = hessian;
                i++;
            }
        }
    }

    /**
     * Copies the gradients that are stored by a vector to a vector of ordinates that may be passed to LAPACK's DSYSV
     * routine.
     *
     * @tparam GradientIterator The type of the iterator that provides access to the gradients
     * @param gradientIterator  An iterator of template type`GradientIterator` that provides random access to the
     *                          gradients that are stored in the vector
     * @param output            A pointer to an array of type `float64`, shape `(n)`, the gradients should be copied to
     * @param n                 The number of ordinates
     */
    template<class GradientIterator>
    static inline void copyOrdinates(GradientIterator gradientIterator, float64* output, uint32 n) {
        for (uint32 i = 0; i < n; i++) {
            float64 gradient = gradientIterator[i];
            output[i] = -gradient;
        }
    }

    static inline float64 calculateExampleWisePredictionInternally(uint32 numPredictions, float64* scores,
                                                                   float64* gradients, float64* hessians,
                                                                   float64 l2RegularizationWeight, Blas& blas,
                                                                   Lapack& lapack, int dsysvLwork,
                                                                   float64* dsysvTmpArray1, int* dsysvTmpArray2,
                                                                   double* dsysvTmpArray3, float64* dspmvTmpArray) {
        // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
        lapack.dsysv(dsysvTmpArray1, dsysvTmpArray2, dsysvTmpArray3, scores, numPredictions, dsysvLwork);

        // Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
        float64 overallQualityScore = blas.ddot(scores, gradients, numPredictions);
        blas.dspmv(hessians, scores, dspmvTmpArray, numPredictions);
        overallQualityScore += 0.5 * blas.ddot(scores, dspmvTmpArray, numPredictions);

        // Add the L2 regularization term to the overall quality score...
        overallQualityScore += 0.5 * l2RegularizationWeight * l2NormPow<float64*>(scores, numPredictions);
        return overallQualityScore;
    }

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rules, as well as corresponding
     * quality scores, based on the gradients and Hessians that have been calculated according to a loss function that
     * is applied example-wise.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<class T>
    class AbstractExampleWiseRuleEvaluation : public IExampleWiseRuleEvaluation {

        protected:

            const T& labelIndices_;

            std::shared_ptr<Lapack> lapackPtr_;

            int dsysvLwork_;

            float64* dsysvTmpArray1_;

            int* dsysvTmpArray2_;

            double* dsysvTmpArray3_;

            float64* dspmvTmpArray_;

            void initializeTmpArrays(uint32 numPredictions) {
                dsysvTmpArray1_ = (float64*) malloc(numPredictions * numPredictions * sizeof(float64));
                dsysvTmpArray2_ = (int*) malloc(numPredictions * sizeof(int));
                dspmvTmpArray_ = (float64*) malloc(numPredictions * sizeof(float64));

                // Query the optimal "lwork" parameter to be used by LAPACK's DSYSV routine...
                dsysvLwork_ = lapackPtr_->queryDsysvLworkParameter(dsysvTmpArray1_, dspmvTmpArray_, numPredictions);
                dsysvTmpArray3_ = (double*) malloc(dsysvLwork_ * sizeof(double));
            }

        public:

            /**
             * @param labelIndices  A reference to an object of template type `T` that provides access to the indices of
             *                      the labels for which the rules may predict
             * @param lapackPtr     A shared pointer to an object of type `Lapack` that allows to execute different
             *                      LAPACK routines
             */
            AbstractExampleWiseRuleEvaluation(const T& labelIndices, std::shared_ptr<Lapack> lapackPtr)
                : labelIndices_(labelIndices), lapackPtr_(lapackPtr), dsysvTmpArray1_(nullptr),
                  dsysvTmpArray2_(nullptr), dsysvTmpArray3_(nullptr), dspmvTmpArray_(nullptr) {

            }

            virtual ~AbstractExampleWiseRuleEvaluation() {
                free(dsysvTmpArray1_);
                free(dsysvTmpArray2_);
                free(dsysvTmpArray3_);
                free(dspmvTmpArray_);
            }

    };

}
