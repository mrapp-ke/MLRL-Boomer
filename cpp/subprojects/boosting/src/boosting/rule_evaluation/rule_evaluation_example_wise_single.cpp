#include "boosting/rule_evaluation/rule_evaluation_example_wise_single.hpp"
#include "common/validation.hpp"


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

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            ExampleWiseSingleLabelRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight)
                : labelIndices_(labelIndices), l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseExampleWiseStatisticVector& statisticVector) override {

            }

            const IScoreVector& calculateExampleWisePrediction(
                DenseExampleWiseStatisticVector& statisticVector) override {

            }

            const IScoreVector& calculatePrediction(const DenseExampleWiseStatisticVector& statisticVector) override {
                // TODO Implement
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
