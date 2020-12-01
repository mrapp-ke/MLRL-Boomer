#include "rule_evaluation_example_wise_binning.h"
#include "rule_evaluation_example_wise_common.h"

using namespace boosting;


/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied example wise using L2 regularization.
 * The labels are assigned to bins based on the corresponding gradients.
 *
 * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
 */
template<class T>
class BinningExampleWiseRuleEvaluation : public AbstractExampleWiseRuleEvaluation<T> {

    private:

        float64 l2RegularizationWeight_;

        uint32 numPositiveBins_;

        uint32 numNegativeBins_;

        std::shared_ptr<Blas> blasPtr_;

    protected:

        void calculateLabelWisePrediction(const DenseExampleWiseStatisticVector& statisticVector,
                                          DenseLabelWiseScoreVector<T>& scoreVector) override {
            // TODO
        }

        void calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector,
                                            DenseScoreVector<T>& scoreVector, int dsysvLwork, float64* dsysvTmpArray1,
                                            int* dsysvTmpArray2, double* dsysvTmpArray3,
                                            float64* dspmvTmpArray) override {
            // TODO
        }

    public:

        /**
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         * @param numPositiveBins           The number of bins to be used for labels that should be predicted
         *                                  positively. Must be at least 1
         * @param numNegativeBins           The number of bins to be used for labels that should be predicted
         *                                  negatively. Must be at least 1
         * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
         *                                  different BLAS routines
         * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
         *                                  different LAPACK routines
         */
        BinningExampleWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight, uint32 numPositiveBins,
                                         uint32 numNegativeBins, std::shared_ptr<Blas> blasPtr,
                                         std::shared_ptr<Lapack> lapackPtr)
            : AbstractExampleWiseRuleEvaluation<T>(labelIndices, lapackPtr),
              l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
              numNegativeBins_(numNegativeBins), blasPtr_(blasPtr) {

        }

};

BinningExampleWiseRuleEvaluationFactory::BinningExampleWiseRuleEvaluationFactory(float64 l2RegularizationWeight,
                                                                                 uint32 numPositiveBins,
                                                                                 uint32 numNegativeBins,
                                                                                 std::shared_ptr<Blas> blasPtr,
                                                                                 std::shared_ptr<Lapack> lapackPtr)
    : l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
      numNegativeBins_(numNegativeBins), blasPtr_(blasPtr), lapackPtr_(lapackPtr) {

}

std::unique_ptr<IExampleWiseRuleEvaluation> BinningExampleWiseRuleEvaluationFactory::create(
        const FullIndexVector& indexVector) const {
    return std::make_unique<BinningExampleWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                               numPositiveBins_, numNegativeBins_,
                                                                               blasPtr_, lapackPtr_);
}

std::unique_ptr<IExampleWiseRuleEvaluation> BinningExampleWiseRuleEvaluationFactory::create(
        const PartialIndexVector& indexVector) const {
    return std::make_unique<BinningExampleWiseRuleEvaluation<PartialIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                                  numPositiveBins_, numNegativeBins_,
                                                                                  blasPtr_, lapackPtr_);
}
