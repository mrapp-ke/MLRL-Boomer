#include "boosting/rule_evaluation/rule_evaluation_label_wise_binning.hpp"
#include "boosting/binning/label_binning_equal_width.hpp"
#include "common/data/arrays.hpp"
#include "common/rule_evaluation/score_vector_label_wise_binned_dense.hpp"
#include "rule_evaluation_label_wise_binning_common.hpp"
#include <cstdlib>


namespace boosting {

    typedef ILabelBinning<DenseLabelWiseStatisticVector::gradient_const_iterator,
                          DenseLabelWiseStatisticVector::hessian_const_iterator> LabelBinning;

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
     * Hessians that are stored by a `DenseLabelWiseStatisticVector` using L2 regularization. The labels are assigned to
     * bins based on the corresponding gradients.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseBinningLabelWiseRuleEvaluation final : public ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            float64 l2RegularizationWeight_;

            uint32 maxBins_;

            std::unique_ptr<LabelBinning> binningPtr_;

            DenseBinnedLabelWiseScoreVector<T> scoreVector_;

            float64* tmpGradients_;

            float64* tmpHessians_;

            uint32* numElementsPerBin_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param maxBins                   The maximum number of bins to assign labels to
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            DenseBinningLabelWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight, uint32 maxBins,
                                                std::unique_ptr<LabelBinning> binningPtr)
                : l2RegularizationWeight_(l2RegularizationWeight), maxBins_(maxBins),
                  binningPtr_(std::move(binningPtr)),
                  scoreVector_(DenseBinnedLabelWiseScoreVector<T>(labelIndices, maxBins + 1)),
                  tmpGradients_((float64*) malloc(maxBins * sizeof(float64))),
                  tmpHessians_((float64*) malloc(maxBins * sizeof(float64))),
                  numElementsPerBin_((uint32*) malloc(maxBins * sizeof(uint32))) {
                // The last bin is used for labels with zero statistics. For this particular bin, the prediction and
                // quality score is always zero.
                scoreVector_.scores_binned_begin()[maxBins] = 0;
                scoreVector_.quality_scores_binned_begin()[maxBins] = 0;
            }

            ~DenseBinningLabelWiseRuleEvaluation() {
                free(tmpGradients_);
                free(tmpHessians_);
                free(numElementsPerBin_);
            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseLabelWiseStatisticVector& statisticVector) override {
                // Obtain information about the bins to be used...
                LabelInfo labelInfo = binningPtr_->getLabelInfo(statisticVector.gradients_cbegin(),
                                                                statisticVector.gradients_cend(),
                                                                statisticVector.hessians_cbegin(),
                                                                statisticVector.hessians_cend(),
                                                                l2RegularizationWeight_);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;
                scoreVector_.setNumBins(numBins, false);

                // Reset arrays to zero...
                setArrayToZeros(tmpGradients_, numBins);
                setArrayToZeros(tmpHessians_, numBins);
                setArrayToZeros(numElementsPerBin_, numBins);

                // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
                auto callback = [this](uint32 binIndex, uint32 labelIndex, float64 gradient, float64 hessian) {
                    tmpGradients_[binIndex] += gradient;
                    tmpHessians_[binIndex] += hessian;
                    numElementsPerBin_[binIndex] += 1;
                    scoreVector_.indices_binned_begin()[labelIndex] = binIndex;
                };
                auto zeroCallback = [this](uint32 labelIndex) {
                    scoreVector_.indices_binned_begin()[labelIndex] = maxBins_;
                };
                binningPtr_->createBins(labelInfo, statisticVector.gradients_cbegin(), statisticVector.gradients_cend(),
                                        statisticVector.hessians_cbegin(), statisticVector.hessians_cend(),
                                        l2RegularizationWeight_, callback, zeroCallback);

                // Compute predictions and quality scores...
                scoreVector_.overallQualityScore = calculateLabelWisePredictionInternally<
                        typename DenseBinnedLabelWiseScoreVector<T>::score_binned_iterator,
                        typename DenseBinnedLabelWiseScoreVector<T>::quality_score_binned_iterator, float64*, float64*,
                        uint32*>(
                    numBins, scoreVector_.scores_binned_begin(), scoreVector_.quality_scores_binned_begin(),
                    tmpGradients_, tmpHessians_, numElementsPerBin_, l2RegularizationWeight_);
                return scoreVector_;
            }

            const IScoreVector& calculatePrediction(const DenseLabelWiseStatisticVector& statisticVector) override {
                // TODO Implement
                return scoreVector_;
            }

    };

    EqualWidthBinningLabelWiseRuleEvaluationFactory::EqualWidthBinningLabelWiseRuleEvaluationFactory(
            float64 l2RegularizationWeight, float32 binRatio, uint32 minBins, uint32 maxBins)
        : l2RegularizationWeight_(l2RegularizationWeight), binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

    }

    std::unique_ptr<ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector>> EqualWidthBinningLabelWiseRuleEvaluationFactory::createDense(
            const CompleteIndexVector& indexVector) const {
        std::unique_ptr<LabelBinning> binningPtr =
            std::make_unique<EqualWidthLabelBinning<DenseLabelWiseStatisticVector::gradient_const_iterator,
                                                    DenseLabelWiseStatisticVector::hessian_const_iterator>>(binRatio_,
                                                                                                            minBins_,
                                                                                                            maxBins_);
        uint32 maxBins = binningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseBinningLabelWiseRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                          l2RegularizationWeight_,
                                                                                          maxBins,
                                                                                          std::move(binningPtr));
    }

    std::unique_ptr<ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector>> EqualWidthBinningLabelWiseRuleEvaluationFactory::createDense(
            const PartialIndexVector& indexVector) const {
        std::unique_ptr<LabelBinning> binningPtr =
            std::make_unique<EqualWidthLabelBinning<DenseLabelWiseStatisticVector::gradient_const_iterator,
                                                    DenseLabelWiseStatisticVector::hessian_const_iterator>>(binRatio_,
                                                                                                            minBins_,
                                                                                                            maxBins_);
        uint32 maxBins = binningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseBinningLabelWiseRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                         l2RegularizationWeight_,
                                                                                         maxBins,
                                                                                         std::move(binningPtr));
    }

}
