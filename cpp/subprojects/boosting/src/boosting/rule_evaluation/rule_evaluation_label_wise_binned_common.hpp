/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "common/data/arrays.hpp"
#include "boosting/data/statistic_vector_label_wise_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"


namespace boosting {

    /**
     * Calculates the score to be predicted for individual bins and returns an overall quality score that assesses the
     * quality of the predictions.
     *
     * @tparam ScoreIterator            The type of the iterator that provides access to the gradients and Hessians
     * @param statisticIterator         An iterator that provides random access to the gradients and Hessians
     * @param scoreIterator             An iterator, the calculated scores should be written to
     * @param weights                   An iterator that provides access to the weights of individual bins
     * @param numElements               The number of bins
     * @param l1RegularizationWeight    The L1 regularization weight
     * @param l2RegularizationWeight    The L2 regularization weight
     * @return                          The overall quality score that has been calculated
     */
    template<typename ScoreIterator>
    static inline float64 calculateBinnedScores(DenseLabelWiseStatisticVector::const_iterator statisticIterator,
                                                ScoreIterator scoreIterator, const uint32* weights, uint32 numElements,
                                                float64 l1RegularizationWeight, float64 l2RegularizationWeight) {
        float64 overallQualityScore = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 weight = weights[i];
            const Tuple<float64>& tuple = statisticIterator[i];
            float64 predictedScore = calculateLabelWiseScore(tuple.first, tuple.second, weight * l1RegularizationWeight,
                                                             weight * l2RegularizationWeight);
            scoreIterator[i] = predictedScore;
            overallQualityScore += calculateLabelWiseQualityScore(predictedScore, tuple.first, tuple.second,
                                                                  weight * l1RegularizationWeight,
                                                                  weight * l2RegularizationWeight);
        }

        return overallQualityScore;
    }

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rules, as well as corresponding
     * quality scores, based on the gradients and Hessians that have been calculated according to a loss function that
     * is applied label-wise and using gradient-based label binning.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam T                The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename T>
    class AbstractLabelWiseBinnedRuleEvaluation : public IRuleEvaluation<StatisticVector> {

        private:

            uint32 maxBins_;

            DenseBinnedScoreVector<T> scoreVector_;

            DenseLabelWiseStatisticVector aggregatedStatisticVector_;

            uint32* numElementsPerBin_;

            float64* criteria_;

            float64 l1RegularizationWeight_;

            float64 l2RegularizationWeight_;

            std::unique_ptr<ILabelBinning> binningPtr_;

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
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            AbstractLabelWiseBinnedRuleEvaluation(const T& labelIndices, float64 l1RegularizationWeight,
                                                  float64 l2RegularizationWeight,
                                                  std::unique_ptr<ILabelBinning> binningPtr)
                : maxBins_(binningPtr->getMaxBins(labelIndices.getNumElements())),
                  scoreVector_(DenseBinnedScoreVector<T>(labelIndices, maxBins_ + 1, true)),
                  aggregatedStatisticVector_(DenseLabelWiseStatisticVector(maxBins_)),
                  numElementsPerBin_(new uint32[maxBins_]), criteria_(new float64[labelIndices.getNumElements()]),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
                  binningPtr_(std::move(binningPtr)) {
                // The last bin is used for labels for which the corresponding criterion is zero. For this particular
                // bin, the prediction is always zero.
                scoreVector_.scores_binned_begin()[maxBins_] = 0;
            }

            virtual ~AbstractLabelWiseBinnedRuleEvaluation() override {
                delete[] numElementsPerBin_;
                delete[] criteria_;
            }

            const IScoreVector& calculatePrediction(StatisticVector& statisticVector) override final {
                // Calculate label-wise criteria...
                uint32 numCriteria = scoreVector_.getNumElements();
                this->calculateLabelWiseCriteria(statisticVector, criteria_, numCriteria, l1RegularizationWeight_,
                                                 l2RegularizationWeight_);

                // Obtain information about the bins to be used...
                LabelInfo labelInfo = binningPtr_->getLabelInfo(criteria_, numCriteria);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;
                scoreVector_.setNumBins(numBins, false);

                // Reset arrays to zero...
                DenseLabelWiseStatisticVector::iterator aggregatedStatisticIterator =
                    aggregatedStatisticVector_.begin();
                setArrayToZeros(aggregatedStatisticIterator, numBins);
                setArrayToZeros(numElementsPerBin_, numBins);

                // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                typename DenseBinnedScoreVector<T>::index_binned_iterator binIndexIterator =
                    scoreVector_.indices_binned_begin();
                auto callback = [=](uint32 binIndex, uint32 labelIndex) {
                    aggregatedStatisticIterator[binIndex] += statisticIterator[labelIndex];
                    numElementsPerBin_[binIndex] += 1;
                    binIndexIterator[labelIndex] = binIndex;
                };
                auto zeroCallback = [=](uint32 labelIndex) {
                    binIndexIterator[labelIndex] = maxBins_;
                };
                binningPtr_->createBins(labelInfo, criteria_, numCriteria, callback, zeroCallback);

                // Compute predictions, as well as an overall quality score...
                typename DenseBinnedScoreVector<T>::score_binned_iterator scoreIterator =
                    scoreVector_.scores_binned_begin();
                scoreVector_.overallQualityScore = calculateBinnedScores(aggregatedStatisticIterator, scoreIterator,
                                                                         numElementsPerBin_, numBins,
                                                                         l1RegularizationWeight_,
                                                                         l2RegularizationWeight_);
                return scoreVector_;
            }

    };

    /**
     * Allows to calculate the predictions of complete rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseLabelWiseStatisticVector` using L1 and L2 regularization. The
     * labels are assigned to bins based on the gradients and Hessians.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseLabelWiseCompleteBinnedRuleEvaluation final :
            public AbstractLabelWiseBinnedRuleEvaluation<DenseLabelWiseStatisticVector, T> {

        protected:

            void calculateLabelWiseCriteria(const DenseLabelWiseStatisticVector& statisticVector, float64* criteria,
                                            uint32 numCriteria, float64 l1RegularizationWeight,
                                            float64 l2RegularizationWeight) override {
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();

                for (uint32 i = 0; i < numCriteria; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    criteria[i] = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight,
                                                          l2RegularizationWeight);
                }
            }

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            DenseLabelWiseCompleteBinnedRuleEvaluation(const T& labelIndices, float64 l1RegularizationWeight,
                                                       float64 l2RegularizationWeight,
                                                       std::unique_ptr<ILabelBinning> binningPtr)
                : AbstractLabelWiseBinnedRuleEvaluation<DenseLabelWiseStatisticVector, T>(labelIndices,
                                                                                          l1RegularizationWeight,
                                                                                          l2RegularizationWeight,
                                                                                          std::move(binningPtr)) {

            }

    };

}
