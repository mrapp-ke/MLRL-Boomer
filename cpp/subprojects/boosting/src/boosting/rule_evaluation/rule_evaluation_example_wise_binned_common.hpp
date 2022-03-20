/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/math/math.hpp"
#include "common/data/arrays.hpp"
#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "rule_evaluation_example_wise_complete_common.hpp"


namespace boosting {

    static inline uint32 removeEmptyBins(uint32* numElementsPerBin, uint32* binIndices, uint32 numBins) {
        uint32 n = 0;

        for (uint32 i = 0; i < numBins; i++) {
            binIndices[i] = n;
            uint32 numElements = numElementsPerBin[i];

            if (numElements > 0) {
                numElementsPerBin[n] = numElements;
                n++;
            }
        }

        return n;
    }

    template<typename BinIndexIterator>
    static inline void aggregateGradientsAndHessians(
            DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator,
            DenseExampleWiseStatisticVector::hessian_const_iterator hessianIterator, uint32 numElements,
            BinIndexIterator binIndexIterator, const uint32* binIndices, float64* gradients, float64* hessians,
            uint32 maxBins) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 originalBinIndex = binIndexIterator[i];

            if (originalBinIndex != maxBins) {
                uint32 binIndex = binIndices[originalBinIndex];
                binIndexIterator[i] = binIndex;

                // Add the gradient that corresponds to the `i`-th element of the original gradient vector to the
                // corresponding element of the aggregated gradient vector...
                gradients[binIndex] += gradientIterator[i];

                // Add the Hessian that corresponds to the `i`-th element on the diagonal of the original Hessian matrix to
                // the corresponding element of the aggregated Hessian matrix...
                hessians[triangularNumber(binIndex + 1) - 1] += hessianIterator[triangularNumber(i + 1) - 1];
            }
        }

        for (uint32 i = 1; i < numElements; i++) {
            uint32 binIndex = binIndexIterator[i];

            if (binIndex != maxBins) {
                for (uint32 j = 0; j < i; j++) {
                    uint32 binIndex2 = binIndexIterator[j];

                    // Add the hessian at the `i`-th row and `j`-th column of the original Hessian matrix to the
                    // corresponding element of the aggregated Hessian matrix, if the labels at indices `i` and `j` do not
                    // belong to the same bin...
                    if (binIndex2 != maxBins && binIndex != binIndex2) {
                        uint32 r, c;

                        if (binIndex < binIndex2) {
                            r = binIndex;
                            c = binIndex2;
                        } else {
                            r = binIndex2;
                            c = binIndex;
                        }

                        hessians[triangularNumber(c) + r] += hessianIterator[triangularNumber(i) + j];
                    }
                }
            }
        }
    }

    static inline void addL1RegularizationWeight(float64* ordinates, uint32 numPredictions, const uint32* weights,
                                                 float64 l1RegularizationWeight) {
        for (uint32 i = 0; i < numPredictions; i++) {
            uint32 weight = weights[i];
            float64 gradient = ordinates[i];
            ordinates[i] += (weight * getL1RegularizationWeight(gradient, l1RegularizationWeight));
        }
    }

    static inline void addL2RegularizationWeight(float64* coefficients, uint32 numPredictions, const uint32* weights,
                                                 float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numPredictions; i++) {
            uint32 weight = weights[i];
            coefficients[(i * numPredictions) + i] += (weight * l2RegularizationWeight);
        }
    }

    /**
     * Calculates and returns the regularization term.
     *
     * @tparam ScoreIterator            The type of the iterator that provides access to the predicted scores
     * @param scores                    An iterator that provides random access to the predicted scores
     * @param numElementsPerBin         A pointer to an array of type `uint32`, shape `(numBins)`, that provides random
     *                                  access to the number of elements per bin
     * @param numBins                   The number of bins
     * @param l1RegularizationWeight    The weight of the L1 regularization term
     * @param l2RegularizationWeight    The weight of the L2 regularization term
     */
    template<typename ScoreIterator>
    static inline float64 calculateRegularizationTerm(ScoreIterator scores, const uint32* numElementsPerBin,
                                                      uint32 numBins, float64 l1RegularizationWeight,
                                                      float64 l2RegularizationWeight) {
        float64 regularizationTerm;

        if (l1RegularizationWeight > 0) {
            regularizationTerm = l1RegularizationWeight * l1Norm(scores, numElementsPerBin, numBins);
        } else {
            regularizationTerm = 0;
        }

        if (l2RegularizationWeight > 0) {
            regularizationTerm += 0.5 * l2RegularizationWeight * l2NormPow(scores, numElementsPerBin, numBins);
        }

        return regularizationTerm;
    }

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rules, as well as an overall
     * quality score, based on the gradients and Hessians that have been calculated according to a loss function that is
     * applied example-wise and using gradient-based label binning.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam T                The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename T>
    class AbstractExampleWiseBinnedRuleEvaluation : public AbstractExampleWiseRuleEvaluation<StatisticVector, T> {

        private:

            uint32 maxBins_;

            DenseBinnedScoreVector<T> scoreVector_;

            float64* aggregatedGradients_;

            float64* aggregatedHessians_;

            uint32* binIndices_;

            uint32* numElementsPerBin_;

            float64* criteria_;

            float64 l1RegularizationWeight_;

            float64 l2RegularizationWeight_;

            std::unique_ptr<ILabelBinning> binningPtr_;

            const Blas& blas_;

            const Lapack& lapack_;

        protected:

            /**
             * Must be implemented by subclasses in order to calculate label-wise criteria that are used to determine
             * the mapping from labels to bins.
             *
             * @param statisticVector           A reference to an object of template type `StatisticVector` that stores
             *                                  the gradients and Hessians
             * @param criteria                  A pointer to an array of type `float64`, shape `(numCriteria)`, the
             *                                  label-wise criteria should be written to
             * @param numCriteria               The number of label-wise criteria to be calculated
             * @param l1RegularizationWeight    The L1 regularization weight
             * @param l2RegularizationWeight    The L2 regularization weight
             */
            virtual void calculateLabelWiseCriteria(const StatisticVector& statisticVector, float64* criteria,
                                                    uint32 numCriteria, float64 l1RegularizationWeight,
                                                    float64 l2RegularizationWeight) = 0;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param maxBins                   The maximum number of bins
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             */
            AbstractExampleWiseBinnedRuleEvaluation(const T& labelIndices, uint32 maxBins,
                                                    float64 l1RegularizationWeight, float64 l2RegularizationWeight,
                                                    std::unique_ptr<ILabelBinning> binningPtr, const Blas& blas,
                                                    const Lapack& lapack)
                : AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, T>(maxBins, lapack),
                  maxBins_(maxBins), scoreVector_(DenseBinnedScoreVector<T>(labelIndices, maxBins + 1, true)),
                  aggregatedGradients_(new float64[maxBins]),
                  aggregatedHessians_(new float64[triangularNumber(maxBins)]), binIndices_(new uint32[maxBins]),
                  numElementsPerBin_(new uint32[maxBins]), criteria_(new float64[labelIndices.getNumElements()]),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
                  binningPtr_(std::move(binningPtr)), blas_(blas), lapack_(lapack) {
                // The last bin is used for labels for which the corresponding criterion is zero. For this particular
                // bin, the prediction is always zero.
                scoreVector_.scores_binned_begin()[maxBins_] = 0;
            }

            virtual ~AbstractExampleWiseBinnedRuleEvaluation() override {
                delete[] aggregatedGradients_;
                delete[] aggregatedHessians_;
                delete[] binIndices_;
                delete[] numElementsPerBin_;
                delete[] criteria_;
            }

            /**
             * @see `IRuleEvaluation::calculatePrediction`
             */
            const IScoreVector& calculatePrediction(DenseExampleWiseStatisticVector& statisticVector) override {
                // Calculate label-wise criteria...
                uint32 numCriteria = scoreVector_.getNumElements();
                this->calculateLabelWiseCriteria(statisticVector, criteria_, numCriteria, l1RegularizationWeight_,
                                                 l2RegularizationWeight_);

                // Obtain information about the bins to be used...
                LabelInfo labelInfo = binningPtr_->getLabelInfo(criteria_, numCriteria);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;

                if (numBins > 0) {
                    // Reset arrays to zero...
                    setArrayToZeros(numElementsPerBin_, numBins);

                    // Apply binning method in order to aggregate the gradients and Hessians that belong to the same
                    // bins...
                    typename DenseBinnedScoreVector<T>::index_binned_iterator binIndexIterator =
                        scoreVector_.indices_binned_begin();
                    auto callback = [=](uint32 binIndex, uint32 labelIndex) {
                        numElementsPerBin_[binIndex] += 1;
                        binIndexIterator[labelIndex] = binIndex;
                    };
                    auto zeroCallback = [=](uint32 labelIndex) {
                        binIndexIterator[labelIndex] = maxBins_;
                    };
                    binningPtr_->createBins(labelInfo, criteria_, numCriteria, callback, zeroCallback);

                    // Determine number of non-empty bins...
                    numBins = removeEmptyBins(numElementsPerBin_, binIndices_, numBins);
                    scoreVector_.setNumBins(numBins, false);

                    // Aggregate gradients and Hessians...
                    setArrayToZeros(aggregatedGradients_, numBins);
                    setArrayToZeros(aggregatedHessians_, triangularNumber(numBins));
                    aggregateGradientsAndHessians(statisticVector.gradients_cbegin(), statisticVector.hessians_cbegin(),
                                                  numCriteria, binIndexIterator, binIndices_, aggregatedGradients_,
                                                  aggregatedHessians_, maxBins_);

                    // Copy Hessians to the matrix of coefficients and add regularization weight to its diagonal...
                    copyCoefficients(aggregatedHessians_, this->dsysvTmpArray1_, numBins);
                    addL2RegularizationWeight(this->dsysvTmpArray1_, numBins, numElementsPerBin_,
                                              l2RegularizationWeight_);

                    // Copy gradients to the vector of ordinates...
                    typename DenseBinnedScoreVector<T>::score_binned_iterator scoreIterator =
                        scoreVector_.scores_binned_begin();
                    copyOrdinates(aggregatedGradients_, scoreIterator, numBins);
                    addL1RegularizationWeight(scoreIterator, numBins, numElementsPerBin_, l1RegularizationWeight_);

                    // Calculate the scores to be predicted for the individual labels by solving a system of linear
                    // equations...
                    lapack_.dsysv(this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_, scoreIterator,
                                  numBins, this->dsysvLwork_);

                    // Calculate the overall quality score...
                    float64 overallQualityScore = calculateOverallQualityScore(scoreIterator, aggregatedGradients_,
                                                                               aggregatedHessians_,
                                                                               this->dspmvTmpArray_, numBins, blas_);

                    // Evaluate regularization term...
                    overallQualityScore += calculateRegularizationTerm(scoreIterator, numElementsPerBin_, numBins,
                                                                       l1RegularizationWeight_,
                                                                       l2RegularizationWeight_);

                    scoreVector_.overallQualityScore = overallQualityScore;
                } else {
                    setArrayToValue(scoreVector_.indices_binned_begin(), numCriteria, maxBins_);
                    scoreVector_.overallQualityScore = 0;
                }

                return scoreVector_;
            }

    };

    /**
     * Allows to calculate the predictions of complete rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseExampleWiseStatisticVector` using L1 and L2 regularization. The
     * labels are assigned to bins based on the gradients and Hessians.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseExampleWiseCompleteBinnedRuleEvaluation final :
            public AbstractExampleWiseBinnedRuleEvaluation<DenseExampleWiseStatisticVector, T> {

        protected:

            void calculateLabelWiseCriteria(const DenseExampleWiseStatisticVector& statisticVector, float64* criteria,
                                            uint32 numCriteria, float64 l1RegularizationWeight,
                                            float64 l2RegularizationWeight) override {
                DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                    statisticVector.gradients_cbegin();
                DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator =
                    statisticVector.hessians_diagonal_cbegin();

                for (uint32 i = 0; i < numCriteria; i++) {
                    criteria[i] = calculateLabelWiseScore(gradientIterator[i], hessianIterator[i],
                                                          l1RegularizationWeight, l2RegularizationWeight);
                }
            }

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param maxBins                   The maximum number of bins
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             */
            DenseExampleWiseCompleteBinnedRuleEvaluation(const T& labelIndices, uint32 maxBins,
                                                         float64 l1RegularizationWeight, float64 l2RegularizationWeight,
                                                         std::unique_ptr<ILabelBinning> binningPtr, const Blas& blas,
                                                         const Lapack& lapack)
                : AbstractExampleWiseBinnedRuleEvaluation<DenseExampleWiseStatisticVector, T>(
                      labelIndices, maxBins, l1RegularizationWeight, l2RegularizationWeight, std::move(binningPtr),
                      blas, lapack) {

            }

    };

}
