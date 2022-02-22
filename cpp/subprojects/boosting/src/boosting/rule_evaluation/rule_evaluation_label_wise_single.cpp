#include "boosting/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"


namespace boosting {

    static inline constexpr float64 calculateLabelWiseQualityScore(float64 gradient, float64 hessian,
                                                                   float64 l1RegularizationWeight,
                                                                   float64 l2RegularizationWeight) {
        float64 l1Weight = getL1RegularizationWeight(gradient, l1RegularizationWeight);
        float64 l1Term = l1Weight != 0
                            ? ((2 * gradient * l1Weight) - (3 * l1RegularizationWeight * l1RegularizationWeight))
                            : (-gradient * l1RegularizationWeight);
        return divideOrZero(-0.5 * (gradient * gradient + l1Term), hessian + l2RegularizationWeight);
    }

    /**
     * Allows to calculate the predictions of single-label rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseLabelWiseStatisticVector` using L1 and L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseLabelWiseSingleLabelRuleEvaluation final : public IRuleEvaluation<DenseLabelWiseStatisticVector> {

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
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DenseLabelWiseSingleLabelRuleEvaluation(const T& labelIndices, float64 l1RegularizationWeight,
                                                    float64 l2RegularizationWeight)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(1)),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_)),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const IScoreVector& calculatePrediction(DenseLabelWiseStatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                const Tuple<float64>& firstTuple = statisticIterator[0];
                float64 bestQualityScore = calculateLabelWiseQualityScore(firstTuple.first, firstTuple.second,
                                                                          l1RegularizationWeight_,
                                                                          l2RegularizationWeight_);
                uint32 bestIndex = 0;

                for (uint32 i = 1; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 qualityScore = calculateLabelWiseQualityScore(tuple.first, tuple.second,
                                                                          l1RegularizationWeight_,
                                                                          l2RegularizationWeight_);

                    if (qualityScore < bestQualityScore) {
                        bestIndex = i;
                        bestQualityScore = qualityScore;
                    }
                }

                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                scoreIterator[0] = calculateLabelWiseScore(statisticIterator[bestIndex].first,
                                                           statisticIterator[bestIndex].second,
                                                           l1RegularizationWeight_, l2RegularizationWeight_);
                indexVector_.begin()[0] = labelIndices_.cbegin()[bestIndex];
                scoreVector_.overallQualityScore = bestQualityScore;
                return scoreVector_;
            }

    };

    /**
     * Allows to calculate the predictions of single-label rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `SparseLabelWiseStatisticVector` using L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class SparseLabelWiseSingleLabelRuleEvaluation final : public IRuleEvaluation<SparseLabelWiseStatisticVector> {

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
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            SparseLabelWiseSingleLabelRuleEvaluation(const T& labelIndices, float64 l1RegularizationWeight,
                                                     float64 l2RegularizationWeight)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(1)),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_)),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const IScoreVector& calculatePrediction(SparseLabelWiseStatisticVector& statisticVector) override {
                float64 sumOfWeights = statisticVector.getSumOfWeights();
                SparseLabelWiseStatisticVector::const_iterator iterator = statisticVector.cbegin();
                SparseLabelWiseStatisticVector::const_iterator end = statisticVector.cend();
                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                PartialIndexVector::iterator indexIterator = indexVector_.begin();

                if (iterator != end) {
                    const IndexedValue<Triple<float64>>& firstEntry = *iterator;
                    const Triple<float64>& firstTriple = firstEntry.value;
                    float64 bestGradient = firstTriple.first;
                    float64 bestHessian = firstTriple.second + (sumOfWeights - firstTriple.third);
                    float64 bestQualityScore = calculateLabelWiseQualityScore(bestGradient, bestHessian,
                                                                              l1RegularizationWeight_,
                                                                              l2RegularizationWeight_);
                    uint32 bestIndex = firstEntry.index;
                    iterator++;

                    for (; iterator != end; iterator++) {
                        const IndexedValue<Triple<float64>>& entry = *iterator;
                        const Triple<float64>& triple = entry.value;
                        float64 gradient = triple.first;
                        float64 hessian = triple.second + (sumOfWeights - triple.third);
                        float64 qualityScore = calculateLabelWiseQualityScore(gradient, hessian,
                                                                              l1RegularizationWeight_,
                                                                              l2RegularizationWeight_);

                        if (qualityScore < bestQualityScore) {
                            bestGradient = gradient;
                            bestHessian = hessian;
                            bestIndex = entry.index;
                            bestQualityScore = qualityScore;
                        }
                    }

                    scoreIterator[0] = calculateLabelWiseScore(bestGradient, bestHessian, l1RegularizationWeight_,
                                                               l2RegularizationWeight_);
                    indexIterator[0] = bestIndex;
                    scoreVector_.overallQualityScore = bestQualityScore;
                } else {
                    scoreIterator[0] = 0;
                    indexIterator[0] = labelIndices_.cbegin()[0];
                    scoreVector_.overallQualityScore = 0;
                }

                return scoreVector_;
            }

    };

    LabelWiseSingleLabelRuleEvaluationFactory::LabelWiseSingleLabelRuleEvaluationFactory(float64 l1RegularizationWeight,
                                                                                         float64 l2RegularizationWeight)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseLabelWiseSingleLabelRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                              l1RegularizationWeight_,
                                                                                              l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseLabelWiseSingleLabelRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                             l1RegularizationWeight_,
                                                                                             l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::create(
            const SparseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        return std::make_unique<SparseLabelWiseSingleLabelRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                               l1RegularizationWeight_,
                                                                                               l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::create(
            const SparseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<SparseLabelWiseSingleLabelRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                              l1RegularizationWeight_,
                                                                                              l2RegularizationWeight_);
    }

}