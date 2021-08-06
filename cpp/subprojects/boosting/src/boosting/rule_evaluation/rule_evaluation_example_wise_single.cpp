#include "boosting/rule_evaluation/rule_evaluation_example_wise_single.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/validation.hpp"
#include "rule_evaluation_label_wise_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of single-label rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseExampleWiseStatisticVector` using L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class ExampleWiseSingleLabelRuleEvaluation final : public IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector> {

        private:

            T labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            ExampleWiseSingleLabelRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(1)),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_)),
                  l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseExampleWiseStatisticVector& statisticVector) override {

            }

            const IScoreVector& calculateExampleWisePrediction(
                DenseExampleWiseStatisticVector& statisticVector) override {

            }

            const IScoreVector& calculatePrediction(const DenseExampleWiseStatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                    statisticVector.gradients_cbegin();
                DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator =
                    statisticVector.hessians_diagonal_cbegin();
                float64 bestScore = calculateLabelWiseScore(gradientIterator[0], hessianIterator[0],
                                                            l2RegularizationWeight_);
                float64 bestAbsScore = std::abs(bestScore);
                uint32 bestIndex = 0;

                for (uint32 i = 1; i < numElements; i++) {
                    float64 score = calculateLabelWiseScore(gradientIterator[i], hessianIterator[i],
                                                            l2RegularizationWeight_);
                    float64 absScore = std::abs(score);

                    if (absScore > bestAbsScore) {
                        bestIndex = i;
                        bestScore = score;
                        bestAbsScore = absScore;
                    }
                }

                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                scoreIterator[0] = bestScore;
                indexVector_.begin()[0] = labelIndices_.cbegin()[bestIndex];
                scoreVector_.overallQualityScore = calculateLabelWiseQualityScore(bestScore,
                                                                                  gradientIterator[bestIndex],
                                                                                  hessianIterator[bestIndex],
                                                                                  l2RegularizationWeight_);
                return scoreVector_;
            }

    };

    ExampleWiseSingleLabelRuleEvaluationFactory::ExampleWiseSingleLabelRuleEvaluationFactory(
            float64 l2RegularizationWeight)
        : l2RegularizationWeight_(l2RegularizationWeight) {
        assertGreaterOrEqual<float64>("l2RegularizationWeight", l2RegularizationWeight, 0);
    }

    std::unique_ptr<IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseSingleLabelRuleEvaluationFactory::createDense(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<ExampleWiseSingleLabelRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                           l2RegularizationWeight_);
    }

    std::unique_ptr<IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseSingleLabelRuleEvaluationFactory::createDense(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<ExampleWiseSingleLabelRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                          l2RegularizationWeight_);
    }

}
