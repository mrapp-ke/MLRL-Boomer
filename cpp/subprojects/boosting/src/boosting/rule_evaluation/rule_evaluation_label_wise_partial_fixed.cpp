#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_fixed.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/math/math.hpp"


namespace boosting {

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
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_)),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const IScoreVector& calculatePrediction(DenseLabelWiseStatisticVector& statisticVector) override {
                // TODO
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
        return std::make_unique<DenseLabelWiseFixedPartialRuleEvaluation<CompleteIndexVector>>(
            indexVector, labelRatio_, minLabels_, maxLabels_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedPartialRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseLabelWiseFixedPartialRuleEvaluation<PartialIndexVector>>(
            indexVector, labelRatio_, minLabels_, maxLabels_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

}
