#include "rule_evaluation_label_wise_binning.h"
#include "rule_evaluation_label_wise_binning_common.h"
#include "../../../common/cpp/data/arrays.h"
#include "../../../common/cpp/rule_evaluation/score_vector_label_wise_binned_dense.h"
#include "../binning/label_binning_equal_width.h"
#include <cstdlib>

using namespace boosting;


/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied label-wise using L2 regularization.
 * The labels are assigned to bins based on the corresponding gradients.
 *
 * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
 */
template<class T>
class BinningLabelWiseRuleEvaluation final : public ILabelWiseRuleEvaluation {

    private:

        float64 l2RegularizationWeight_;

        uint32 maxBins_;

        std::unique_ptr<ILabelBinning<DenseLabelWiseStatisticVector>> binningPtr_;

        DenseBinnedLabelWiseScoreVector<T> scoreVector_;

        float64* tmpGradients_;

        float64* tmpHessians_;

        uint32* numElementsPerBin_;

    public:

        /**
         * @param labelIndices              A reference to an object of template type `T` that provides access to the
         *                                  indices of the labels for which rules may predict
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         * @param maxBins                   The maximum number of bins to assign labels to
         * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be used
         *                                  to assign labels to bins
         */
        BinningLabelWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight, uint32 maxBins,
                                       std::unique_ptr<ILabelBinning<DenseLabelWiseStatisticVector>> binningPtr)
            : l2RegularizationWeight_(l2RegularizationWeight), maxBins_(maxBins), binningPtr_(std::move(binningPtr)),
              scoreVector_(DenseBinnedLabelWiseScoreVector<T>(labelIndices, maxBins + 1)),
              tmpGradients_((float64*) malloc(maxBins * sizeof(float64))),
              tmpHessians_((float64*) malloc(maxBins * sizeof(float64))),
              numElementsPerBin_((uint32*) malloc(maxBins * sizeof(uint32))) {
            // The last bin is used for labels with zero statistics. For this particular bin, the prediction and quality
            // score is always zero.
            scoreVector_.scores_binned_begin()[maxBins] = 0;
            scoreVector_.quality_scores_binned_begin()[maxBins] = 0;
        }

        ~BinningLabelWiseRuleEvaluation() {
            free(tmpGradients_);
            free(tmpHessians_);
            free(numElementsPerBin_);
        }

        const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseLabelWiseStatisticVector& statisticVector) override {
            // Obtain information about the bins to be used...
            LabelInfo labelInfo = binningPtr_->getLabelInfo(statisticVector);
            uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;
            scoreVector_.setNumBins(numBins, false);

            // Reset arrays to zero...
            setArrayToZeros(tmpGradients_, numBins);
            setArrayToZeros(tmpHessians_, numBins);
            setArrayToZeros(numElementsPerBin_, numBins);

            // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
            auto callback = [this, &statisticVector](uint32 binIndex, uint32 labelIndex, float64 statistic) {
                tmpGradients_[binIndex] += statistic;
                float64 hessian = statisticVector.hessians_cbegin()[labelIndex];
                tmpHessians_[binIndex] += hessian;
                numElementsPerBin_[binIndex] += 1;
                scoreVector_.indices_binned_begin()[labelIndex] = binIndex;
            };
            auto zeroCallback = [this](uint32 labelIndex) {
                scoreVector_.indices_binned_begin()[labelIndex] = maxBins_;
            };
            binningPtr_->createBins(labelInfo, statisticVector, callback, zeroCallback);

            // Compute predictions and quality scores...
            scoreVector_.overallQualityScore = calculateLabelWisePredictionInternally<
                    typename DenseBinnedLabelWiseScoreVector<T>::score_binned_iterator,
                    typename DenseBinnedLabelWiseScoreVector<T>::quality_score_binned_iterator, float64*, float64*,
                    uint32*>(
                numBins, scoreVector_.scores_binned_begin(), scoreVector_.quality_scores_binned_begin(), tmpGradients_,
                tmpHessians_, numElementsPerBin_, l2RegularizationWeight_);
            return scoreVector_;
        }

};

EqualWidthBinningLabelWiseRuleEvaluationFactory::EqualWidthBinningLabelWiseRuleEvaluationFactory(
        float64 l2RegularizationWeight, float32 binRatio, uint32 minBins, uint32 maxBins)
    : l2RegularizationWeight_(l2RegularizationWeight), binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

}

std::unique_ptr<ILabelWiseRuleEvaluation> EqualWidthBinningLabelWiseRuleEvaluationFactory::create(
        const FullIndexVector& indexVector) const {
    std::unique_ptr<ILabelBinning<DenseLabelWiseStatisticVector>> binningPtr =
        std::make_unique<EqualWidthLabelBinning<DenseLabelWiseStatisticVector>>(binRatio_, minBins_, maxBins_);
    uint32 maxBins = binningPtr->getMaxBins(indexVector.getNumElements());
    return std::make_unique<BinningLabelWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                             maxBins, std::move(binningPtr));
}

std::unique_ptr<ILabelWiseRuleEvaluation> EqualWidthBinningLabelWiseRuleEvaluationFactory::create(
        const PartialIndexVector& indexVector) const {
    std::unique_ptr<ILabelBinning<DenseLabelWiseStatisticVector>> binningPtr =
        std::make_unique<EqualWidthLabelBinning<DenseLabelWiseStatisticVector>>(binRatio_, minBins_, maxBins_);
    uint32 maxBins = binningPtr->getMaxBins(indexVector.getNumElements());
    return std::make_unique<BinningLabelWiseRuleEvaluation<PartialIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                                maxBins, std::move(binningPtr));
}
