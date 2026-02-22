/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"
#include "mlrl/common/data/array.hpp"
#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"
#include "mlrl/common/util/iterators.hpp"
#include "rule_evaluation_decomposable_common.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * Calculates the score to be predicted for individual bins and returns the overall quality of the predictions.
     *
     * @tparam GradientIterator         The type of the iterator that provides access to the gradients
     * @tparam HessianIterator          The type of the iterator that provides access to the Hessians
     * @tparam ScoreIterator            The type of the iterator, the calculated scores should be written to
     * @param statisticIterator         An iterator that provides random access to the gradients and Hessians
     * @param scoreIterator             An iterator, the calculated scores should be written to
     * @param weights                   An iterator to the weights of individual bins
     * @param numElements               The number of bins
     * @param l1RegularizationWeight    The L1 regularization weight
     * @param l2RegularizationWeight    The L2 regularization weight
     * @return                          The overall quality that has been calculated
     */
    template<typename GradientIterator, typename HessianIterator, typename ScoreIterator>
    static inline typename util::iterator_value<GradientIterator> calculateBinnedScores(
      GradientIterator gradientIterator, HessianIterator hessianIterator, ScoreIterator scoreIterator,
      View<uint32>::const_iterator weights, uint32 numElements, float32 l1RegularizationWeight,
      float32 l2RegularizationWeight) {
        using statistic_type = util::iterator_value<ScoreIterator>;
        statistic_type quality = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 weight = weights[i];
            statistic_type predictedScore =
              calculateOutputWiseScore(gradientIterator[i], hessianIterator[i], weight * l1RegularizationWeight,
                                       weight * l2RegularizationWeight);
            scoreIterator[i] = predictedScore;
            quality += calculateOutputWiseQuality(predictedScore, gradientIterator[i], hessianIterator[i],
                                                  weight * l1RegularizationWeight, weight * l2RegularizationWeight);
        }

        return quality;
    }

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rules, as well as their overall
     * quality, based on the gradients and Hessians that have been calculated according to a decomposable loss function
     * and using gradient-based label binning.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the labels for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class AbstractDecomposableBinnedRuleEvaluation : public IRuleEvaluation<StatisticVector> {
        private:

            using statistic_type = StatisticVector::statistic_type;

            const uint32 maxBins_;

            DenseBinnedScoreVector<statistic_type, IndexVector> scoreVector_;

            DenseVector<statistic_type> aggregatedGradientVector_;

            DenseVector<statistic_type> aggregatedHessianVector_;

            Array<uint32> numElementsPerBin_;

            Array<statistic_type> criteria_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

            const std::unique_ptr<ILabelBinning<statistic_type>> binningPtr_;

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
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            AbstractDecomposableBinnedRuleEvaluation(const IndexVector& labelIndices, bool indicesSorted,
                                                     float32 l1RegularizationWeight, float32 l2RegularizationWeight,
                                                     std::unique_ptr<ILabelBinning<statistic_type>> binningPtr)
                : maxBins_(binningPtr->getMaxBins(labelIndices.getNumElements())),
                  scoreVector_(labelIndices, maxBins_ + 1, indicesSorted), aggregatedGradientVector_(maxBins_),
                  aggregatedHessianVector_(maxBins_), numElementsPerBin_(maxBins_),
                  criteria_(labelIndices.getNumElements()), l1RegularizationWeight_(l1RegularizationWeight),
                  l2RegularizationWeight_(l2RegularizationWeight), binningPtr_(std::move(binningPtr)) {
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
                scoreVector_.setNumBins(numBins, false);

                // Reset arrays to zero...
                typename DenseVector<statistic_type>::iterator aggregatedGradientIterator =
                  aggregatedGradientVector_.begin();
                typename DenseVector<statistic_type>::iterator aggregatedHessianIterator =
                  aggregatedHessianVector_.begin();
                Array<uint32>::iterator numElementsPerBinIterator = numElementsPerBin_.begin();
                std::fill(aggregatedGradientIterator, aggregatedGradientIterator + numBins, (statistic_type) 0);
                std::fill(aggregatedHessianIterator, aggregatedHessianIterator + numBins, (statistic_type) 0);
                std::fill(numElementsPerBinIterator, numElementsPerBinIterator + numBins, 0);

                // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
                typename StatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();
                typename StatisticVector::hessian_const_iterator hessianIterator = statisticVector.hessians_cbegin();
                typename DenseBinnedScoreVector<statistic_type, IndexVector>::bin_index_iterator binIndexIterator =
                  scoreVector_.bin_indices_begin();
                auto callback = [=](uint32 binIndex, uint32 labelIndex) {
                    aggregatedGradientIterator[binIndex] += gradientIterator[labelIndex];
                    aggregatedHessianIterator[binIndex] += hessianIterator[labelIndex];
                    numElementsPerBinIterator[binIndex] += 1;
                    binIndexIterator[labelIndex] = binIndex;
                };
                auto zeroCallback = [=, this](uint32 labelIndex) {
                    binIndexIterator[labelIndex] = maxBins_;
                };
                binningPtr_->createBins(labelInfo, criteria_.cbegin(), numCriteria, callback, zeroCallback);

                // Compute predictions, as well as their overall quality...
                typename DenseBinnedScoreVector<statistic_type, IndexVector>::bin_value_iterator binValueIterator =
                  scoreVector_.bin_values_begin();
                scoreVector_.quality = calculateBinnedScores(aggregatedGradientIterator, aggregatedHessianIterator,
                                                             binValueIterator, numElementsPerBin_.cbegin(), numBins,
                                                             l1RegularizationWeight_, l2RegularizationWeight_);
                return scoreVector_;
            }
    };

    /**
     * Allows to calculate the predictions of complete rules, as well as their overall quality, based on the gradients
     * Hessians that are stored by a vector using L1 and L2 regularization. The labels are assigned to bins based on the
     * gradients and Hessians.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the labels for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DecomposableCompleteBinnedRuleEvaluation final
        : public AbstractDecomposableBinnedRuleEvaluation<StatisticVector, IndexVector> {
        private:

            using statistic_type = StatisticVector::statistic_type;

        protected:

            uint32 calculateOutputWiseCriteria(const StatisticVector& statisticVector,
                                               typename View<statistic_type>::iterator criteria, uint32 numCriteria,
                                               float32 l1RegularizationWeight,
                                               float32 l2RegularizationWeight) override {
                typename StatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();
                typename StatisticVector::hessian_const_iterator hessianIterator = statisticVector.hessians_cbegin();

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
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            DecomposableCompleteBinnedRuleEvaluation(const IndexVector& labelIndices, float32 l1RegularizationWeight,
                                                     float32 l2RegularizationWeight,
                                                     std::unique_ptr<ILabelBinning<statistic_type>> binningPtr)
                : AbstractDecomposableBinnedRuleEvaluation<StatisticVector, IndexVector>(
                    labelIndices, true, l1RegularizationWeight, l2RegularizationWeight, std::move(binningPtr)) {}
    };

}
