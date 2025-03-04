/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/util/math.hpp"
#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"
#include "rule_evaluation_non_decomposable_complete_common.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * Removes empty bins from an array that keeps track of the number of elements per bin, as well as an array that
     * stores the index of each bin.
     *
     * @param numElementsPerBin An iterator to the number of elements per bin
     * @param binIndices        An iterator to the indices of each bin
     * @param numBins           The number of available bins
     */
    static inline uint32 removeEmptyBins(View<uint32>::iterator numElementsPerBin, View<uint32>::iterator binIndices,
                                         uint32 numBins) {
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

    /**
     * Aggregates the gradients and Hessians of all elements that have been assigned to the same bin.
     *
     * @tparam GradientIterator             The type of the iterator that provides access to the gradients
     * @tparam HessianIterator              The type of the iterator that provides access to the Hessians
     * @tparam AggregatedGradientIterator   The type of the iterator, the aggregated gradients should be written to
     * @tparam AggregatedHessianIterator    The type of the iterator, the aggregated Hessians should be written to
     * @param gradientIterator              An iterator that provides random access to the gradients
     * @param hessianIterator               An iterator that provides random access to the Hessians
     * @param numElements                   The total number of available elements
     * @param binIndexIterator              An iterator that provides random access to the indices of the bins
     *                                      individual elements have been assigned to
     * @param binIndices                    An iterator to the indices of each bin
     * @param gradients                     An iterator, the aggregated gradients should be written to
     * @param hessians                      An iterator, the aggregated Hessians should be written to
     * @param maxBins                       The maximum number of bins
     */
    template<typename GradientIterator, typename HessianIterator, typename AggregatedGradientIterator,
             typename AggregatedHessianIterator>
    static inline void aggregateGradientsAndHessians(GradientIterator gradients, HessianIterator hessians,
                                                     uint32 numElements, View<uint32>::iterator binIndexIterator,
                                                     View<uint32>::const_iterator binIndices,
                                                     AggregatedGradientIterator aggregatedGradients,
                                                     AggregatedHessianIterator aggregatedHessians, uint32 maxBins) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 originalBinIndex = binIndexIterator[i];

            if (originalBinIndex != maxBins) {
                uint32 binIndex = binIndices[originalBinIndex];
                binIndexIterator[i] = binIndex;

                // Add the gradient that corresponds to the `i`-th element of the original gradient vector to the
                // corresponding element of the aggregated gradient vector...
                aggregatedGradients[binIndex] += gradients[i];

                // Add the Hessian that corresponds to the `i`-th element on the diagonal of the original Hessian matrix
                // to the corresponding element of the aggregated Hessian matrix...
                aggregatedHessians[util::triangularNumber(binIndex + 1) - 1] +=
                  hessians[util::triangularNumber(i + 1) - 1];
            }
        }

        for (uint32 i = 1; i < numElements; i++) {
            uint32 binIndex = binIndexIterator[i];

            if (binIndex != maxBins) {
                for (uint32 j = 0; j < i; j++) {
                    uint32 binIndex2 = binIndexIterator[j];

                    // Add the hessian at the `i`-th row and `j`-th column of the original Hessian matrix to the
                    // corresponding element of the aggregated Hessian matrix, if the labels at indices `i` and `j` do
                    // not belong to the same bin...
                    if (binIndex2 != maxBins && binIndex != binIndex2) {
                        uint32 r, c;

                        if (binIndex < binIndex2) {
                            r = binIndex;
                            c = binIndex2;
                        } else {
                            r = binIndex2;
                            c = binIndex;
                        }

                        aggregatedHessians[util::triangularNumber(c) + r] += hessians[util::triangularNumber(i) + j];
                    }
                }
            }
        }
    }

    /**
     * Adds a L1 regularization weight to a vector of ordinates.
     *
     * @tparam OrdinateIterator         The type of the iterator that provides access to the ordinates
     * @param ordinates                 An iterator, the L1 regularization weight should be added to
     * @param numPredictions            The number of ordinates
     * @param weights                   An iterator to the weight of each ordinate
     * @param l1RegularizationWeight    The L1 regularization weight to be added to the ordinates
     */
    template<typename OrdinateIterator>
    static inline void addL1RegularizationWeight(OrdinateIterator ordinates, uint32 numPredictions,
                                                 View<uint32>::const_iterator weights, float32 l1RegularizationWeight) {
        for (uint32 i = 0; i < numPredictions; i++) {
            uint32 weight = weights[i];
            typename util::iterator_value<OrdinateIterator> gradient = ordinates[i];
            ordinates[i] += (weight * getL1RegularizationWeight(gradient, l1RegularizationWeight));
        }
    }

    /**
     * Adds a L2 regularization weight to the diagonal of a matrix of coefficients.
     *
     * @tparam CoefficientIterator      The type of the iterator that provides access to the coefficients
     * @param coefficients              An iterator, the regularization weight should be added to
     * @param numPredictions            The number of coefficients on the diagonal
     * @param weights                   An iterator to the weight of each coefficient
     * @param l2RegularizationWeight    The L2 regularization weight to be added to the coefficients
     */
    template<typename CoefficientIterator>
    static inline void addL2RegularizationWeight(CoefficientIterator coefficients, uint32 numPredictions,
                                                 View<uint32>::const_iterator weights, float32 l2RegularizationWeight) {
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
     * @param numElementsPerBin         An iterator to the number of elements per bin
     * @param numBins                   The number of bins
     * @param l1RegularizationWeight    The weight of the L1 regularization term
     * @param l2RegularizationWeight    The weight of the L2 regularization term
     */
    template<typename ScoreIterator>
    static inline typename util::iterator_value<ScoreIterator> calculateRegularizationTerm(
      ScoreIterator scores, View<uint32>::const_iterator numElementsPerBin, uint32 numBins,
      float32 l1RegularizationWeight, float32 l2RegularizationWeight) {
        typename util::iterator_value<ScoreIterator> regularizationTerm;

        if (l1RegularizationWeight > 0) {
            regularizationTerm = l1RegularizationWeight * util::l1Norm(scores, numElementsPerBin, numBins);
        } else {
            regularizationTerm = 0;
        }

        if (l2RegularizationWeight > 0) {
            regularizationTerm += 0.5 * l2RegularizationWeight * util::l2NormPow(scores, numElementsPerBin, numBins);
        }

        return regularizationTerm;
    }

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rules, as well as their overall
     * quality, based on the gradients and Hessians that have been calculated according to a non-decomposable loss
     * function and using gradient-based label binning.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the labels for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class AbstractNonDecomposableBinnedRuleEvaluation
        : public AbstractNonDecomposableRuleEvaluation<StatisticVector, IndexVector> {
        private:

            typedef typename StatisticVector::statistic_type statistic_type;

            const uint32 maxBins_;

            DenseBinnedScoreVector<statistic_type, IndexVector> scoreVector_;

            Array<statistic_type> aggregatedGradients_;

            Array<statistic_type> aggregatedHessians_;

            Array<uint32> binIndices_;

            Array<uint32> numElementsPerBin_;

            Array<statistic_type> criteria_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

            const std::unique_ptr<ILabelBinning<statistic_type>> binningPtr_;

            const std::unique_ptr<Blas<statistic_type>> blasPtr_;

            const std::unique_ptr<Lapack<statistic_type>> lapackPtr_;

        protected:

            /**
             * Must be implemented by subclasses in order to calculate output-wise criteria that are used to determine
             * the mapping from outputs to bins.
             *
             * @param statisticVector           A reference to an object of template type `StatisticVector` that stores
             *                                  the gradients and Hessians
             * @param criteria                  An iterator, the output-wise criteria should be written to
             * @param numCriteria               The number of output-wise criteria to be calculated
             * @param l1RegularizationWeight    The L1 regularization weight
             * @param l2RegularizationWeight    The L2 regularization weight
             * @return                          The number of output-wise criteria that have been calculated
             */
            virtual uint32 calculateOutputWiseCriteria(const StatisticVector& statisticVector,
                                                       typename View<statistic_type>::iterator criteria,
                                                       uint32 numCriteria, float32 l1RegularizationWeight,
                                                       float32 l2RegularizationWeight) = 0;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param indicesSorted             True, if the given indices are guaranteed to be sorted, false otherwise
             * @param maxBins                   The maximum number of bins
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             * @param blasPtr                   An unique pointer to an object of type `Blas` that allows to execute
             *                                  BLAS routines
             * @param lapackPtr                 An unique pointer to an object of type `Lapack` that allows to execute
             *                                  LAPACK routines
             */
            AbstractNonDecomposableBinnedRuleEvaluation(const IndexVector& labelIndices, bool indicesSorted,
                                                        uint32 maxBins, float32 l1RegularizationWeight,
                                                        float32 l2RegularizationWeight,
                                                        std::unique_ptr<ILabelBinning<statistic_type>> binningPtr,
                                                        std::unique_ptr<Blas<statistic_type>> blasPtr,
                                                        std::unique_ptr<Lapack<statistic_type>> lapackPtr)
                : AbstractNonDecomposableRuleEvaluation<StatisticVector, IndexVector>(maxBins, *lapackPtr),
                  maxBins_(maxBins), scoreVector_(labelIndices, maxBins + 1, indicesSorted),
                  aggregatedGradients_(maxBins), aggregatedHessians_(util::triangularNumber(maxBins)),
                  binIndices_(maxBins), numElementsPerBin_(maxBins), criteria_(labelIndices.getNumElements()),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
                  binningPtr_(std::move(binningPtr)), blasPtr_(std::move(blasPtr)), lapackPtr_(std::move(lapackPtr)) {
                // The last bin is used for labels for which the corresponding criterion is zero. For this particular
                // bin, the prediction is always zero.
                scoreVector_.bin_values_begin()[maxBins_] = 0;
            }

            /**
             * @see `IRuleEvaluation::evaluate`
             */
            const IScoreVector& calculateScores(StatisticVector& statisticVector) override final {
                // Calculate label-wise criteria...
                uint32 numCriteria =
                  this->calculateOutputWiseCriteria(statisticVector, criteria_.begin(), scoreVector_.getNumElements(),
                                                    l1RegularizationWeight_, l2RegularizationWeight_);

                // Obtain information about the bins to be used...
                LabelInfo<statistic_type> labelInfo = binningPtr_->getLabelInfo(criteria_.cbegin(), numCriteria);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;

                if (numBins > 0) {
                    // Reset arrays to zero...
                    util::setViewToZeros(numElementsPerBin_.begin(), numBins);

                    // Apply binning method in order to aggregate the gradients and Hessians that belong to the same
                    // bins...
                    typename DenseBinnedScoreVector<statistic_type, IndexVector>::bin_index_iterator binIndexIterator =
                      scoreVector_.bin_indices_begin();
                    auto callback = [=, this](uint32 binIndex, uint32 labelIndex) {
                        numElementsPerBin_[binIndex] += 1;
                        binIndexIterator[labelIndex] = binIndex;
                    };
                    auto zeroCallback = [=, this](uint32 labelIndex) {
                        binIndexIterator[labelIndex] = maxBins_;
                    };
                    binningPtr_->createBins(labelInfo, criteria_.cbegin(), numCriteria, callback, zeroCallback);

                    // Determine number of non-empty bins...
                    numBins = removeEmptyBins(numElementsPerBin_.begin(), binIndices_.begin(), numBins);
                    scoreVector_.setNumBins(numBins, false);

                    // Aggregate gradients and Hessians...
                    util::setViewToZeros(aggregatedGradients_.begin(), numBins);
                    util::setViewToZeros(aggregatedHessians_.begin(), util::triangularNumber(numBins));
                    aggregateGradientsAndHessians(statisticVector.gradients_cbegin(), statisticVector.hessians_cbegin(),
                                                  numCriteria, binIndexIterator, binIndices_.cbegin(),
                                                  aggregatedGradients_.begin(), aggregatedHessians_.begin(), maxBins_);

                    // Copy Hessians to the matrix of coefficients and add regularization weight to its diagonal...
                    copyCoefficients(aggregatedHessians_.cbegin(), this->sysvTmpArray1_.begin(), numBins);
                    addL2RegularizationWeight(this->sysvTmpArray1_.begin(), numBins, numElementsPerBin_.cbegin(),
                                              l2RegularizationWeight_);

                    // Copy gradients to the vector of ordinates...
                    typename DenseBinnedScoreVector<statistic_type, IndexVector>::bin_value_iterator binValueIterator =
                      scoreVector_.bin_values_begin();
                    copyOrdinates(aggregatedGradients_.cbegin(), binValueIterator, numBins);
                    addL1RegularizationWeight(binValueIterator, numBins, numElementsPerBin_.cbegin(),
                                              l1RegularizationWeight_);

                    // Calculate the scores to be predicted for the individual labels by solving a system of linear
                    // equations...
                    lapackPtr_->sysv(this->sysvTmpArray1_.begin(), this->sysvTmpArray2_.begin(),
                                     this->sysvTmpArray3_.begin(), binValueIterator, numBins, this->sysvLwork_);

                    // Calculate the overall quality...
                    statistic_type quality = calculateOverallQuality(binValueIterator, aggregatedGradients_.begin(),
                                                                     aggregatedHessians_.begin(),
                                                                     this->spmvTmpArray_.begin(), numBins, *blasPtr_);

                    // Evaluate regularization term...
                    quality += calculateRegularizationTerm(binValueIterator, numElementsPerBin_.cbegin(), numBins,
                                                           l1RegularizationWeight_, l2RegularizationWeight_);

                    scoreVector_.quality = quality;
                } else {
                    util::setViewToValue(scoreVector_.bin_indices_begin(), numCriteria, maxBins_);
                    scoreVector_.quality = 0;
                }

                return scoreVector_;
            }
    };

    /**
     * Allows to calculate the predictions of complete rules, as well as their overall quality, based on the gradients
     * and Hessians that are stored by a `DenseNonDecomposableStatisticVector` using L1 and L2 regularization. The
     * labels are assigned to bins based on the gradients and Hessians.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DenseNonDecomposableCompleteBinnedRuleEvaluation final
        : public AbstractNonDecomposableBinnedRuleEvaluation<StatisticVector, IndexVector> {
        private:

            typedef typename StatisticVector::statistic_type statistic_type;

        protected:

            uint32 calculateOutputWiseCriteria(const StatisticVector& statisticVector,
                                               typename View<statistic_type>::iterator criteria, uint32 numCriteria,
                                               float32 l1RegularizationWeight,
                                               float32 l2RegularizationWeight) override {
                typename StatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();
                typename StatisticVector::hessian_diagonal_const_iterator hessianIterator =
                  statisticVector.hessians_diagonal_cbegin();

                for (uint32 i = 0; i < numCriteria; i++) {
                    criteria[i] = calculateOutputWiseScore(gradientIterator[i], hessianIterator[i],
                                                           l1RegularizationWeight, l2RegularizationWeight);
                }

                return numCriteria;
            }

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param maxBins                   The maximum number of bins
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             * @param blasPtr                   An unique pointer to an object of type `Blas` that allows to execute
             *                                  BLAS routines
             * @param lapackPtr                 An unique pointer to an object of type `Lapack` that allows to execute
             *                                  LAPACK routines
             */
            DenseNonDecomposableCompleteBinnedRuleEvaluation(const IndexVector& labelIndices, uint32 maxBins,
                                                             float32 l1RegularizationWeight,
                                                             float32 l2RegularizationWeight,
                                                             std::unique_ptr<ILabelBinning<statistic_type>> binningPtr,
                                                             std::unique_ptr<Blas<statistic_type>> blasPtr,
                                                             std::unique_ptr<Lapack<statistic_type>> lapackPtr)
                : AbstractNonDecomposableBinnedRuleEvaluation<StatisticVector, IndexVector>(
                    labelIndices, true, maxBins, l1RegularizationWeight, l2RegularizationWeight, std::move(binningPtr),
                    std::move(blasPtr), std::move(lapackPtr)) {}
    };

}
