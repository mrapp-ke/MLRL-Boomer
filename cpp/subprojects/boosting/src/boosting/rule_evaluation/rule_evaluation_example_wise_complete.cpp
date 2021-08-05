#include "boosting/rule_evaluation/rule_evaluation_example_wise_complete.hpp"
#include "common/validation.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of complete rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseExampleWiseStatisticVector` using L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class ExampleWiseCompleteRuleEvaluation final : public IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector> {

        private:

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            ExampleWiseCompleteRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight)
                : l2RegularizationWeight_(l2RegularizationWeight) {

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

    ExampleWiseCompleteRuleEvaluationFactory::ExampleWiseCompleteRuleEvaluationFactory(
            float64 l2RegularizationWeight, std::unique_ptr<Blas> blasPtr, std::unique_ptr<Lapack> lapackPtr)
        : l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(std::move(blasPtr)),
          lapackPtr_(std::move(lapackPtr)) {
        assertGreaterOrEqual<float64>("l2RegularizationWeight", l2RegularizationWeight, 0);
        assertNotNull("blasPtr", blasPtr_.get());
        assertNotNull("lapackPtr", lapackPtr_.get());
    }

    std::unique_ptr<IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseCompleteRuleEvaluationFactory::createDense(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<ExampleWiseCompleteRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                        l2RegularizationWeight_);
    }

    std::unique_ptr<IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseCompleteRuleEvaluationFactory::createDense(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<ExampleWiseCompleteRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                       l2RegularizationWeight_);
    }

}
