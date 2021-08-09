#include "boosting/rule_evaluation/rule_evaluation_label_wise_complete_binned.hpp"
#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "common/data/arrays.hpp"
#include "common/validation.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of complete rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseLabelWiseStatisticVector` using L2 regularization. The labels
     * are assigned to bins based on the gradients and Hessians.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class LabelWiseCompleteBinnedRuleEvaluation final : public ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            uint32 maxBins_;

            DenseBinnedScoreVector<T> scoreVector_;

            float64* tmpGradients_;

            float64* tmpHessians_;

            uint32* numElementsPerBin_;

            float64 l2RegularizationWeight_;

            std::unique_ptr<ILabelBinning> binningPtr_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            LabelWiseCompleteBinnedRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight,
                                                  std::unique_ptr<ILabelBinning> binningPtr)
                : maxBins_(binningPtr->getMaxBins(labelIndices.getNumElements())),
                  scoreVector_(DenseBinnedScoreVector<T>(labelIndices, maxBins_ + 1)),
                  tmpGradients_(new float64[maxBins_]), tmpHessians_(new float64[maxBins_]),
                  numElementsPerBin_(new uint32[maxBins_]), l2RegularizationWeight_(l2RegularizationWeight),
                  binningPtr_(std::move(binningPtr)) {
                // The last bin is used for labels for which the corresponding criterion is zero. For this particular
                // bin, the prediction is always zero.
                scoreVector_.scores_binned_begin()[maxBins_] = 0;
            }

            ~LabelWiseCompleteBinnedRuleEvaluation() {
                delete[] tmpGradients_;
                delete[] tmpHessians_;
                delete[] numElementsPerBin_;
            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseLabelWiseStatisticVector& statisticVector) override {

            }

            const IScoreVector& calculatePrediction(const DenseLabelWiseStatisticVector& statisticVector) override {
                // Obtain information about the bins to be used...
                LabelInfo labelInfo = binningPtr_->getLabelInfo(statisticVector, l2RegularizationWeight_);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;
                scoreVector_.setNumBins(numBins, false);

                // Reset arrays to zero...
                setArrayToZeros(tmpGradients_, numBins);
                setArrayToZeros(tmpHessians_, numBins);
                setArrayToZeros(numElementsPerBin_, numBins);

                // TODO Implement
            }

    };

    LabelWiseCompleteBinnedRuleEvaluationFactory::LabelWiseCompleteBinnedRuleEvaluationFactory(
            float64 l2RegularizationWeight, std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : l2RegularizationWeight_(l2RegularizationWeight), labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {
        assertGreaterOrEqual<float64>("l2RegularizationWeight", l2RegularizationWeight, 0);
        assertNotNull("labelBinningFactoryPtr", labelBinningFactoryPtr_.get());
    }

    std::unique_ptr<ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseCompleteBinnedRuleEvaluationFactory::createDense(
            const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<LabelWiseCompleteBinnedRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                            l2RegularizationWeight_,
                                                                                            std::move(labelBinningPtr));
    }

    std::unique_ptr<ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseCompleteBinnedRuleEvaluationFactory::createDense(
            const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<LabelWiseCompleteBinnedRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                           l2RegularizationWeight_,
                                                                                           std::move(labelBinningPtr));
    }

}