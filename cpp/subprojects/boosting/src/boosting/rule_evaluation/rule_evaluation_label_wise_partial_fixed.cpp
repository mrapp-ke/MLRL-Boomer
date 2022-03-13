#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_fixed.hpp"
#include "common/data/indexed_value.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/math/math.hpp"
#include "rule_evaluation_label_wise_common.hpp"
#include <queue>


namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for predefined subset of labels, as well as an
     * overall quality score, based on the gradients and Hessians that are stored by a `DenseLabelWiseStatisticVector`
     * using L1 and L2 regularization.
     */
    class DenseLabelWiseSubsetRuleEvaluation final : public IRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            DenseScoreVector<PartialIndexVector> scoreVector_;

            float64 l1RegularizationWeight_;

            float64 l2RegularizationWeight_;

        public:

            DenseLabelWiseSubsetRuleEvaluation(const PartialIndexVector& labelIndices, float64 l1RegularizationWeight,
                                               float64 l2RegularizationWeight)
                : scoreVector_(DenseScoreVector<PartialIndexVector>(labelIndices, true)),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const IScoreVector& calculatePrediction(DenseLabelWiseStatisticVector& statisticVector) override {
                uint32 numElements = scoreVector_.getNumElements();
                DenseScoreVector<PartialIndexVector>::index_const_iterator indexIterator =
                    scoreVector_.indices_cbegin();
                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                float64 overallQualityScore = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    const Tuple<float64> tuple = statisticIterator[index];
                    float64 predictedScore = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight_,
                                                                     l2RegularizationWeight_);
                    scoreIterator[i] = predictedScore;
                    overallQualityScore += calculateLabelWiseQualityScore(predictedScore, tuple.first, tuple.second,
                                                                          l1RegularizationWeight_,
                                                                          l2RegularizationWeight_);
                }

                scoreVector_.overallQualityScore = overallQualityScore;
                return scoreVector_;
            }

    };

    /**
     * Allows to calculate the predictions of partial rules that predict for a predefined number of labels, as well as
     * an overall quality score, based on the gradients and Hessians that are stored by a
     * `DenseLabelWiseStatisticVector` using L1 and L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseLabelWiseFixedPartialRuleEvaluation final : public IRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            const T& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            float64 l1RegularizationWeight_;

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param labelRatio                A percentage that specifies for how many labels the rule heads should
             *                                  predict, e.g., if 100 labels are available, a percentage of 0.5 means
             *                                  that the rule heads predict for a subset of `ceil(0.5 * 100) = 50`
             *                                  labels. Must be in (0, 1)
             * @param minLabels                 The minimum number of labels for which the rule heads should predict.
             *                                  Must be at least 2
             * @param maxLabels                 The maximum number of labels for which the rule heads should predict.
             *                                  Must be at least `minLabels` or 0, if the maximum number of labels
             *                                  should not be restricted
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DenseLabelWiseFixedPartialRuleEvaluation(const T& labelIndices, float32 labelRatio, uint32 minLabels,
                                                     uint32 maxLabels, float64 l1RegularizationWeight,
                                                     float64 l2RegularizationWeight)
                : labelIndices_(labelIndices),
                  indexVector_(PartialIndexVector(calculateBoundedFraction(labelIndices.getNumElements(), labelRatio,
                                                                           minLabels, maxLabels))),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_, false)),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const IScoreVector& calculatePrediction(DenseLabelWiseStatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                typename T::const_iterator labelIndexIterator = labelIndices_.cbegin();
                uint32 limit = indexVector_.getNumElements();
                std::priority_queue<IndexedValue<float64>, std::vector<IndexedValue<float64>>,
                                    IndexedValue<float64>::Compare> priorityQueue;

                for (uint32 i = 0; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 qualityScore = calculateLabelWiseQualityScore(tuple.first, tuple.second,
                                                                          l1RegularizationWeight_,
                                                                          l2RegularizationWeight_);

                    if (priorityQueue.size() < limit) {
                        priorityQueue.emplace(labelIndexIterator[i], qualityScore);
                    } else if (priorityQueue.top().value > qualityScore) {
                        priorityQueue.pop();
                        priorityQueue.emplace(labelIndexIterator[i], qualityScore);
                    }
                }

                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                float64 overallQualityScore = 0;

                for (uint32 i = 0; i < limit; i++) {
                    const IndexedValue<float64> entry = priorityQueue.top();
                    priorityQueue.pop();
                    uint32 index = entry.index;
                    indexIterator[i] = index;
                    const Tuple<float64>& tuple = statisticIterator[index];
                    scoreIterator[i] = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight_,
                                                               l2RegularizationWeight_);
                    overallQualityScore += entry.value;
                }

                scoreVector_.overallQualityScore = overallQualityScore;
                return scoreVector_;
            }

    };

    LabelWiseFixedPartialRuleEvaluationFactory::LabelWiseFixedPartialRuleEvaluationFactory(
            float32 labelRatio, uint32 minLabels, uint32 maxLabels, float64 l1RegularizationWeight,
            float64 l2RegularizationWeight)
        : labelRatio_(labelRatio), minLabels_(minLabels), maxLabels_(maxLabels),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedPartialRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseLabelWiseFixedPartialRuleEvaluation<CompleteIndexVector>>(indexVector, labelRatio_,
                                                                                               minLabels_, maxLabels_,
                                                                                               l1RegularizationWeight_,
                                                                                               l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedPartialRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseLabelWiseSubsetRuleEvaluation>(indexVector, l1RegularizationWeight_,
                                                                    l2RegularizationWeight_);
    }

}
