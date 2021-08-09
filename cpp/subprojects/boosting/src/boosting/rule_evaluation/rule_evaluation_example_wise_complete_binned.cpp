#include "boosting/rule_evaluation/rule_evaluation_example_wise_complete_binned.hpp"
#include "common/validation.hpp"
#include "rule_evaluation_example_wise_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of complete rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseExampleWiseStatisticVector` using L2 regularization. The labels
     * are assigned to bins based on the gradients and Hessians.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class ExampleWiseCompleteBinnedRuleEvaluation final : public AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, T> {

        private:

            float64 l2RegularizationWeight_;

            std::unique_ptr<ILabelBinning> binningPtr_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             * @param blas                      A reference to an object of type `Blas` that allows to execute different
             *                                  BLAS routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute
             *                                  different LAPACK routines
             */
            ExampleWiseCompleteBinnedRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight,
                                                    std::unique_ptr<ILabelBinning> binningPtr, const Blas& blas,
                                                    const Lapack& lapack)
                : AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, T>(labelIndices, lapack),
                  l2RegularizationWeight_(l2RegularizationWeight), binningPtr_(std::move(binningPtr)), blas_(blas),
                  lapack_(lapack) {

            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseExampleWiseStatisticVector& statisticVector) override {

            }

            const IScoreVector& calculateExampleWisePrediction(
                DenseExampleWiseStatisticVector& statisticVector) override {

            }

            const IScoreVector& calculatePrediction(DenseExampleWiseStatisticVector& statisticVector) override {
                // TODO Implement
            }

    };

    ExampleWiseCompleteBinnedRuleEvaluationFactory::ExampleWiseCompleteBinnedRuleEvaluationFactory(
            float64 l2RegularizationWeight, std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr,
            std::unique_ptr<Blas> blasPtr, std::unique_ptr<Lapack> lapackPtr)
        : l2RegularizationWeight_(l2RegularizationWeight), labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)),
          blasPtr_(std::move(blasPtr)), lapackPtr_(std::move(lapackPtr)) {
        assertGreaterOrEqual<float64>("l2RegularizationWeight", l2RegularizationWeight, 0);
        assertNotNull("labelBinningFactoryPtr", labelBinningFactoryPtr_.get());
        assertNotNull("blasPtr", blasPtr_.get());
        assertNotNull("lapackPtr", lapackPtr_.get());
    }

    std::unique_ptr<IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseCompleteBinnedRuleEvaluationFactory::createDense(
            const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<ExampleWiseCompleteBinnedRuleEvaluation<CompleteIndexVector>>(
            indexVector, l2RegularizationWeight_, std::move(labelBinningPtr), *blasPtr_, *lapackPtr_);
    }

    std::unique_ptr<IExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseCompleteBinnedRuleEvaluationFactory::createDense(
            const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<ExampleWiseCompleteBinnedRuleEvaluation<PartialIndexVector>>(
            indexVector, l2RegularizationWeight_, std::move(labelBinningPtr), *blasPtr_, *lapackPtr_);
    }

}
