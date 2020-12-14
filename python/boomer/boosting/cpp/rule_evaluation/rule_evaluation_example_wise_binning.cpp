#include "rule_evaluation_example_wise_binning.h"
#include "rule_evaluation_label_wise_binning_common.h"
#include "rule_evaluation_example_wise_common.h"
#include "../../../common/cpp/data/vector_mapping_dense.h"
#include "../../../common/cpp/rule_evaluation/score_vector_label_wise_binned_dense.h"
#include "../binning/label_binning_equal_width.h"
#include "../math/blas.h"
#include <cstdlib>

using namespace boosting;


static inline uint32 aggregateGradientsAndHessians(const DenseExampleWiseStatisticVector& statisticVector,
                                                   const DenseMappingVector<uint32>& mapping, float64* gradients,
                                                   float64* hessians, uint32 numBins) {
    DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();
    DenseExampleWiseStatisticVector::hessian_const_iterator hessianIterator = statisticVector.hessians_cbegin();
    DenseMappingVector<uint32>::const_iterator mappingIterator = mapping.cbegin();
    uint32 n = 0;

    for (uint32 i = 0; i < numBins; i++) {
        const DenseMappingVector<uint32>::Entry& bin1 = mappingIterator[i];
        DenseMappingVector<uint32>::Entry::const_iterator it1 = bin1.cbegin();
        DenseMappingVector<uint32>::Entry::const_iterator end1 = bin1.cend();

        if (it1 != end1) {
            float64 sumOfGradients = 0;
            uint32 offset = triangularNumber(n);

            for (; it1 != end1; it1++) {
                uint32 index1 = *it1;
                sumOfGradients += gradientIterator[index1];
                uint32 n2 = 0;

                for (uint32 j = 0; j < i + 1; j++) {
                    const DenseMappingVector<uint32>::Entry& bin2 = mappingIterator[j];
                    DenseMappingVector<uint32>::Entry::const_iterator it2 = bin2.cbegin();
                    DenseMappingVector<uint32>::Entry::const_iterator end2 = bin2.cend();

                    if (it2 != end2) {
                        float64 sumOfHessians = 0;

                        for (; it2 != end2; it2++) {
                            uint32 index2 = *it2;
                            uint32 r, c;

                             if (index1 < index2) {
                                r = index1;
                                c = index2;
                            } else {
                                r = index2;
                                c = index1;
                            }

                            sumOfHessians += hessianIterator[triangularNumber(c) + r];
                        }

                        hessians[offset + n2] = sumOfHessians;
                        n2++;
                    }
                }
            }

            gradients[n] = sumOfGradients;
            n++;
        }
    }

    return n;
}

/**
 * Adds a specific L2 regularization weight to the diagonal of a coefficient matrix.
 *
 * @param output                    A pointer to an array of type `float64`, shape `(n, n)` that stores the coefficients
 * @param n                         The number of rows and columns in the coefficient matrix
 * @param numElementsPerBin         A pointer to an array of type `uint32`, shape `(n)` that stores the number of
 *                                  elements per bin
 * @param l2RegularizationWeight    The L2 regularization weight to be added
 */
static inline void addRegularizationWeight(float64* output, uint32 n, const uint32* numElementsPerBin,
                                           float64 l2RegularizationWeight) {
    for (uint32 i = 0; i < n; i++) {
        float64 weight = (float64) numElementsPerBin[i];
        output[(i * n) + i] += (weight * l2RegularizationWeight);
    }
}

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
                    ruleEvaluation_.mapping_->begin()[binIndex].push_front(originalIndex);
                    ruleEvaluation_.numElementsPerBin_[binIndex] += 1;
                    ruleEvaluation_.scoreVector_->indices_binned_begin()[originalIndex] = binIndex;
                }

        };

        float64 l2RegularizationWeight_;

        uint32 maxBins_;

        std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr_;

        std::shared_ptr<Blas> blasPtr_;

        DenseBinnedScoreVector<T>* scoreVector_;

        DenseBinnedLabelWiseScoreVector<T>* labelWiseScoreVector_;

        float64* tmpGradients_;

        float64* tmpHessians_;

        uint32* numElementsPerBin_;

        DenseMappingVector<uint32>* mapping_;

        IBinningObserver<float64>* binningObserver_;

        const DenseExampleWiseStatisticVector* currentStatisticVector_;

    public:

        /**
         * @param labelIndices              A reference to an object of template type `T` that provides access to the
         *                                  indices of the labels for which the rules may predict
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         * @param maxBins                   The maximum number of bins to assign labels to
         * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be used
         *                                  to assign labels to bins
         * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
         *                                  different BLAS routines
         * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
         *                                  different LAPACK routines
         */
        BinningExampleWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight, uint32 maxBins,
                                         std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr,
                                         std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
            : AbstractExampleWiseRuleEvaluation<T>(labelIndices, lapackPtr),
              l2RegularizationWeight_(l2RegularizationWeight), maxBins_(maxBins), binningPtr_(std::move(binningPtr)),
              blasPtr_(blasPtr), scoreVector_(nullptr), labelWiseScoreVector_(nullptr), tmpGradients_(nullptr),
              tmpHessians_(nullptr), numElementsPerBin_(nullptr), mapping_(nullptr), binningObserver_(nullptr) {

        }

        ~BinningExampleWiseRuleEvaluation() {
            delete scoreVector_;
            delete labelWiseScoreVector_;
            free(tmpGradients_);
            free(tmpHessians_);
            free(numElementsPerBin_);
            delete mapping_;
            delete binningObserver_;
        }

        const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseExampleWiseStatisticVector& statisticVector) override {
            if (labelWiseScoreVector_ == nullptr) {
                labelWiseScoreVector_ = new DenseBinnedLabelWiseScoreVector<T>(this->labelIndices_, maxBins_);
                tmpGradients_ = (float64*) malloc(maxBins_ * sizeof(float64));
                tmpHessians_ = (float64*) malloc(maxBins_ * sizeof(float64));
                numElementsPerBin_ = (uint32*) malloc(maxBins_ * sizeof(uint32));
                binningObserver_ = new BinningExampleWiseRuleEvaluation<T>::LabelWiseBinningObserver(*this);
            }

            // Obtain information about the bins to be used...
            LabelInfo labelInfo = binningPtr_->getLabelInfo(statisticVector);
            uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;
            labelWiseScoreVector_->setNumBins(numBins, false);

            // Reset gradients and Hessians to zero...
            for (uint32 i = 0; i < numBins; i++) {
                tmpGradients_[i] = 0;
                tmpHessians_[i] = 0;
                numElementsPerBin_[i] = 0;
            }

            // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
            currentStatisticVector_ = &statisticVector;
            binningPtr_->createBins(labelInfo, statisticVector, *binningObserver_);

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
                scoreVector_ = new DenseBinnedScoreVector<T>(this->labelIndices_, maxBins_);
                this->initializeTmpArrays(maxBins_);
                tmpGradients_ = (float64*) malloc(maxBins_ * sizeof(float64));
                tmpHessians_ = (float64*) malloc(triangularNumber(maxBins_) * sizeof(float64));
                numElementsPerBin_ = (uint32*) malloc(maxBins_ * sizeof(uint32));
                mapping_ = new DenseMappingVector<uint32>(maxBins_);
                binningObserver_ = new BinningExampleWiseRuleEvaluation<T>::ExampleWiseBinningObserver(*this);
            }

            // Obtain information about the bins to be used...
            LabelInfo labelInfo = binningPtr_->getLabelInfo(statisticVector);
            uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;

            // Reset mapping and the number of elements per bin...
            mapping_->clear();

            for (uint32 i = 0; i < numBins; i++) {
                numElementsPerBin_[i] = 0;
            }

            // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
            currentStatisticVector_ = &statisticVector;
            binningPtr_->createBins(labelInfo, statisticVector, *binningObserver_);
            numBins = aggregateGradientsAndHessians(statisticVector, *mapping_, tmpGradients_, tmpHessians_, numBins);
            scoreVector_->setNumBins(numBins, false);

            typename DenseBinnedScoreVector<T>::score_binned_iterator scoreIterator =
                scoreVector_->scores_binned_begin();
            copyCoefficients<float64*>(tmpHessians_, this->dsysvTmpArray1_, numBins);
            addRegularizationWeight(this->dsysvTmpArray1_, numBins, numElementsPerBin_, l2RegularizationWeight_);
            copyOrdinates<float64*>(tmpGradients_, scoreIterator, numBins);

            // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
            this->lapackPtr_->dsysv(this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_, scoreIterator,
                                    numBins, this->dsysvLwork_);

            // Calculate the overall quality score...
            float64 qualityScore = calculateExampleWiseQualityScore(numBins, scoreIterator, tmpGradients_, tmpHessians_,
                                                                    *blasPtr_, this->dspmvTmpArray_);
            qualityScore += 0.5 * l2RegularizationWeight_ *
                            l2NormPow<typename DenseBinnedScoreVector<T>::score_binned_iterator, uint32*>(
                                scoreIterator, numElementsPerBin_, numBins);
            scoreVector_->overallQualityScore = qualityScore;
            return *scoreVector_;
        }

};

EqualWidthBinningExampleWiseRuleEvaluationFactory::EqualWidthBinningExampleWiseRuleEvaluationFactory(
        float64 l2RegularizationWeight, float32 binRatio, std::shared_ptr<Blas> blasPtr,
        std::shared_ptr<Lapack> lapackPtr)
    : l2RegularizationWeight_(l2RegularizationWeight), binRatio_(binRatio), blasPtr_(blasPtr), lapackPtr_(lapackPtr) {

}

std::unique_ptr<IExampleWiseRuleEvaluation> EqualWidthBinningExampleWiseRuleEvaluationFactory::create(
        const FullIndexVector& indexVector) const {
    std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr =
        std::make_unique<EqualWidthLabelBinning<DenseExampleWiseStatisticVector>>(binRatio_);
    uint32 maxBins = binningPtr->getMaxBins(indexVector.getNumElements());
    return std::make_unique<BinningExampleWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                               maxBins, std::move(binningPtr), blasPtr_,
                                                                               lapackPtr_);
}

std::unique_ptr<IExampleWiseRuleEvaluation> EqualWidthBinningExampleWiseRuleEvaluationFactory::create(
        const PartialIndexVector& indexVector) const {
    std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr =
        std::make_unique<EqualWidthLabelBinning<DenseExampleWiseStatisticVector>>(binRatio_);
    uint32 maxBins = binningPtr->getMaxBins(indexVector.getNumElements());
    return std::make_unique<BinningExampleWiseRuleEvaluation<PartialIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                                  maxBins, std::move(binningPtr),
                                                                                  blasPtr_, lapackPtr_);
}
