#include "rule_evaluation_label_wise_binning.h"
#include "rule_evaluation_label_wise_common.h"
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
class BinningLabelWiseRuleEvaluation : public ILabelWiseRuleEvaluation, public IBinningObserver<float64> {

    private:

        float64 l2RegularizationWeight_;

        uint32 numPositiveBins_;

        uint32 numNegativeBins_;

        std::unique_ptr<ILabelBinning<DenseLabelWiseStatisticVector>> binningPtr_;

        DenseBinnedLabelWiseScoreVector<T> scoreVector_;

        float64* tmpGradients_;

        float64* tmpHessians_;

        const DenseLabelWiseStatisticVector* currentStatisticVector_;

    public:

        /**
         * @param numPositiveBins           The number of bins to be used for labels that should be predicted
         *                                  positively. Must be at least 1
         * @param numNegativeBins           The number of bins to be used for labels that should be predicted
         *                                  negatively. Must be at least 1
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be used
         *                                  to assign labels to bins
         */
        BinningLabelWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight, uint32 numPositiveBins,
                                       uint32 numNegativeBins,
                                       std::unique_ptr<ILabelBinning<DenseLabelWiseStatisticVector>> binningPtr)
            : l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
              numNegativeBins_(numNegativeBins), binningPtr_(std::move(binningPtr)),
              scoreVector_(DenseBinnedLabelWiseScoreVector<T>(labelIndices, numPositiveBins + numNegativeBins)),
              tmpGradients_((float64*) malloc(scoreVector_.getNumBins() * sizeof(float64))),
              tmpHessians_((float64*) malloc(scoreVector_.getNumBins() * sizeof(float64))) {

        }

        ~BinningLabelWiseRuleEvaluation() {
            free(tmpGradients_);
            free(tmpHessians_);
        }

        const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseLabelWiseStatisticVector& statisticVector) override {
            // Reset gradients and Hessians to zero...
            uint32 numBins = scoreVector_.getNumBins();

            for (uint32 i = 0; i < numBins; i++) {
                tmpGradients_[i] = 0;
                tmpHessians_[i] = 0;
            }

            // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
            currentStatisticVector_ = &statisticVector;
            binningPtr_->createBins(numPositiveBins_, numNegativeBins_, statisticVector, *this);

            // Compute predictions and quality scores...
            scoreVector_.overallQualityScore = calculateLabelWisePredictionInternally<
                    typename DenseBinnedLabelWiseScoreVector<T>::score_binned_iterator,
                    typename DenseBinnedLabelWiseScoreVector<T>::quality_score_binned_iterator, float64*, float64*>(
                numBins, scoreVector_.scores_binned_begin(), scoreVector_.quality_scores_binned_begin(), tmpGradients_,
                tmpHessians_, l2RegularizationWeight_);
            return scoreVector_;
        }

        void onBinUpdate(uint32 binIndex, uint32 originalIndex, float64 value) override {
            tmpGradients_[binIndex] += value;
            tmpHessians_[binIndex] += currentStatisticVector_->hessians_cbegin()[originalIndex];
            scoreVector_.indices_binned_begin()[originalIndex] = binIndex;
        }

};

EqualWidthBinningLabelWiseRuleEvaluationFactory::EqualWidthBinningLabelWiseRuleEvaluationFactory(
        float64 l2RegularizationWeight, uint32 numPositiveBins, uint32 numNegativeBins)
    : l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
      numNegativeBins_(numNegativeBins) {

}

std::unique_ptr<ILabelWiseRuleEvaluation> EqualWidthBinningLabelWiseRuleEvaluationFactory::create(
        const FullIndexVector& indexVector) const {
    std::unique_ptr<ILabelBinning<DenseLabelWiseStatisticVector>> binningPtr =
        std::make_unique<EqualWidthLabelBinning<DenseLabelWiseStatisticVector>>();
    return std::make_unique<BinningLabelWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                             numPositiveBins_, numNegativeBins_,
                                                                             std::move(binningPtr));
}

std::unique_ptr<ILabelWiseRuleEvaluation> EqualWidthBinningLabelWiseRuleEvaluationFactory::create(
        const PartialIndexVector& indexVector) const {
    std::unique_ptr<ILabelBinning<DenseLabelWiseStatisticVector>> binningPtr =
        std::make_unique<EqualWidthLabelBinning<DenseLabelWiseStatisticVector>>();
    return std::make_unique<BinningLabelWiseRuleEvaluation<PartialIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                                numPositiveBins_, numNegativeBins_,
                                                                                std::move(binningPtr));
}
