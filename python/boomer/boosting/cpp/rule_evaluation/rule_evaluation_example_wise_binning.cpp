#include "rule_evaluation_example_wise_binning.h"
#include "rule_evaluation_label_wise_binning_common.h"
#include "rule_evaluation_example_wise_common.h"
#include "../../../common/cpp/rule_evaluation/score_vector_label_wise_binned_dense.h"
#include "../binning/label_binning_equal_width.h"
#include "../math/blas.h"
#include <cstdlib>

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

        class LabelWiseBinningObserver : public IBinningObserver<float64> {

            private:

                const BinningExampleWiseRuleEvaluation<T>& ruleEvaluation_;

            public:

                LabelWiseBinningObserver(const BinningExampleWiseRuleEvaluation<T>& ruleEvaluation)
                    : ruleEvaluation_(ruleEvaluation) {

                }

                void onBinUpdate(uint32 binIndex, uint32 originalIndex, float64 value) override {
                    ruleEvaluation_.tmpGradients_[binIndex] += value;
                    float64 hessian =
                        ruleEvaluation_.currentStatisticVector_->hessians_diagonal_cbegin()[originalIndex];
                    ruleEvaluation_.tmpHessians_[binIndex] += hessian;
                    ruleEvaluation_.numElementsPerBin_[binIndex] += 1;
                    ruleEvaluation_.labelWiseScoreVector_->indices_binned_begin()[originalIndex] = binIndex;
                }

        };

        class ExampleWiseBinningObserver : public IBinningObserver<float64> {

            private:

                const BinningExampleWiseRuleEvaluation<T>& ruleEvaluation_;

            public:

                ExampleWiseBinningObserver(const BinningExampleWiseRuleEvaluation<T>& ruleEvaluation)
                    : ruleEvaluation_(ruleEvaluation) {

                }

                void onBinUpdate(uint32 binIndex, uint32 originalIndex, float64 value) override {
                    // TODO
                }

        };

        float64 l2RegularizationWeight_;

        uint32 numPositiveBins_;

        uint32 numNegativeBins_;

        std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr_;

        std::shared_ptr<Blas> blasPtr_;

        DenseBinnedScoreVector<T>* scoreVector_;

        DenseBinnedLabelWiseScoreVector<T>* labelWiseScoreVector_;

        float64* tmpGradients_;

        float64* tmpHessians_;

        uint32* numElementsPerBin_;

        IBinningObserver<float64>* binningObserver_;

        const DenseExampleWiseStatisticVector* currentStatisticVector_;

    public:

        /**
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         * @param numPositiveBins           The number of bins to be used for labels that should be predicted
         *                                  positively. Must be at least 1
         * @param numNegativeBins           The number of bins to be used for labels that should be predicted
         *                                  negatively. Must be at least 1
         * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be used
         *                                  to assign labels to bins
         * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
         *                                  different BLAS routines
         * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
         *                                  different LAPACK routines
         */
        BinningExampleWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight, uint32 numPositiveBins,
                                         uint32 numNegativeBins,
                                         std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr,
                                         std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
            : AbstractExampleWiseRuleEvaluation<T>(labelIndices, lapackPtr),
              l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
              numNegativeBins_(numNegativeBins), binningPtr_(std::move(binningPtr)), blasPtr_(blasPtr),
              scoreVector_(nullptr), labelWiseScoreVector_(nullptr), tmpGradients_(nullptr), tmpHessians_(nullptr),
              numElementsPerBin_(nullptr), binningObserver_(nullptr) {

        }

        ~BinningExampleWiseRuleEvaluation() {
            delete scoreVector_;
            delete labelWiseScoreVector_;
            free(tmpGradients_);
            free(tmpHessians_);
            free(numElementsPerBin_);
            delete binningObserver_;
        }

        const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseExampleWiseStatisticVector& statisticVector) override {
            uint32 numBins;

            if (labelWiseScoreVector_ == nullptr) {
                numBins = numPositiveBins_ + numNegativeBins_;
                labelWiseScoreVector_ = new DenseBinnedLabelWiseScoreVector<T>(this->labelIndices_, numBins);
                tmpGradients_ = (float64*) malloc(numBins * sizeof(float64));
                tmpHessians_ = (float64*) malloc(numBins * sizeof(float64));
                numElementsPerBin_ = (uint32*) malloc(numBins * sizeof(uint32));
                binningObserver_ = new BinningExampleWiseRuleEvaluation<T>::LabelWiseBinningObserver(*this);
            } else {
                numBins = labelWiseScoreVector_->getNumBins();
            }

            // Reset gradients and Hessians to zero...
            for (uint32 i = 0; i < numBins; i++) {
                tmpGradients_[i] = 0;
                tmpHessians_[i] = 0;
                numElementsPerBin_[i] = 0;
            }

            // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
            currentStatisticVector_ = &statisticVector;
            binningPtr_->createBins(numPositiveBins_, numNegativeBins_, statisticVector, *binningObserver_);

            // Compute predictions and quality scores...
            labelWiseScoreVector_->overallQualityScore = calculateLabelWisePredictionInternally<
                    typename DenseBinnedLabelWiseScoreVector<T>::score_binned_iterator,
                    typename DenseBinnedLabelWiseScoreVector<T>::quality_score_binned_iterator, float64*, float64*,
                    uint32*>(
                numBins, labelWiseScoreVector_->scores_binned_begin(),
                labelWiseScoreVector_->quality_scores_binned_begin(), tmpGradients_, tmpHessians_, numElementsPerBin_,
                l2RegularizationWeight_);
            return *labelWiseScoreVector_;
        }

        const IScoreVector& calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector) override {
            if (scoreVector_ == nullptr) {
                uint32 numBins = numPositiveBins_ + numNegativeBins_;
                scoreVector_ = new DenseBinnedScoreVector<T>(this->labelIndices_, numBins);
                this->initializeTmpArrays(numBins);
                binningObserver_ = new BinningExampleWiseRuleEvaluation<T>::ExampleWiseBinningObserver(*this);
            }

            // Reset gradients and Hessians to zero...
            // TODO

            // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
            currentStatisticVector_ = &statisticVector;
            binningPtr_->createBins(numPositiveBins_, numNegativeBins_, statisticVector, *binningObserver_);

            // Compute predictions and quality scores...
            // TODO

            return *scoreVector_;
        }

};

EqualWidthBinningExampleWiseRuleEvaluationFactory::EqualWidthBinningExampleWiseRuleEvaluationFactory(
        float64 l2RegularizationWeight, uint32 numPositiveBins, uint32 numNegativeBins, std::shared_ptr<Blas> blasPtr,
        std::shared_ptr<Lapack> lapackPtr)
    : l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
      numNegativeBins_(numNegativeBins), blasPtr_(blasPtr), lapackPtr_(lapackPtr) {

}

std::unique_ptr<IExampleWiseRuleEvaluation> EqualWidthBinningExampleWiseRuleEvaluationFactory::create(
        const FullIndexVector& indexVector) const {
    std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr =
        std::make_unique<EqualWidthLabelBinning<DenseExampleWiseStatisticVector>>();
    return std::make_unique<BinningExampleWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                               numPositiveBins_, numNegativeBins_,
                                                                               std::move(binningPtr), blasPtr_,
                                                                               lapackPtr_);
}

std::unique_ptr<IExampleWiseRuleEvaluation> EqualWidthBinningExampleWiseRuleEvaluationFactory::create(
        const PartialIndexVector& indexVector) const {
    std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr =
        std::make_unique<EqualWidthLabelBinning<DenseExampleWiseStatisticVector>>();
    return std::make_unique<BinningExampleWiseRuleEvaluation<PartialIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                                  numPositiveBins_, numNegativeBins_,
                                                                                  std::move(binningPtr), blasPtr_,
                                                                                  lapackPtr_);
}
