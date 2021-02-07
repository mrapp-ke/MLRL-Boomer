#include "rule_evaluation_example_wise_binning.hpp"
#include "rule_evaluation_label_wise_binning_common.hpp"
#include "rule_evaluation_example_wise_common.hpp"
#include "../../../common/cpp/data/arrays.hpp"
#include "../../../common/cpp/data/vector_mapping_dense.hpp"
#include "../../../common/cpp/rule_evaluation/score_vector_label_wise_binned_dense.hpp"
#include "../binning/label_binning_equal_width.hpp"
#include "../math/blas.hpp"
#include <cstdlib>


namespace boosting {

    template<class T>
    static inline uint32 aggregateGradientsAndHessians(const DenseExampleWiseStatisticVector& statisticVector,
                                                       const DenseMappingVector<uint32>& mapping,
                                                       uint32* numElementsPerBin,
                                                       DenseBinnedScoreVector<T>& scoreVector, float64* gradients,
                                                       float64* hessians, uint32 numBins) {
        DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();
        DenseExampleWiseStatisticVector::hessian_const_iterator hessianIterator = statisticVector.hessians_cbegin();
        DenseMappingVector<uint32>::const_iterator mappingIterator = mapping.cbegin();
        typename DenseBinnedScoreVector<T>::index_binned_iterator binIndexIterator = scoreVector.indices_binned_begin();
        uint32 binIndex = 0;
        uint32 n = 0;

        // Iterate the bins in increasing order...
        for (uint32 i = 0; i < numBins; i++) {
            const DenseMappingVector<uint32>::Entry& bin = mappingIterator[i];
            DenseMappingVector<uint32>::Entry::const_iterator end = bin.cend();

            // Ignore empty bins...
            if (bin.cbegin() != end) {
                uint32 numElements = 0;
                float64 aggregatedGradients = 0;
                float64 aggregatedDiagonalHessians = 0;

                for (auto it = bin.cbegin(); it != end; it++) {
                    uint32 labelIndex = *it;
                    numElements++;

                    // Map the label at index `labelIndex` to the bin at index `binIndex`...
                    binIndexIterator[labelIndex] = binIndex;

                    // Add the gradient that corresponds to the label at index `labelIndex`...
                    aggregatedGradients += gradientIterator[labelIndex];

                    // Add the Hessian on the diagonal of the original Hessian matrix that corresponds to the label at
                    // index `labelIndex`...
                    aggregatedDiagonalHessians += hessianIterator[triangularNumber(labelIndex + 1) - 1];
                }

                // Iterate the column of the aggregated Hessian matrix that corresponds to the current bin (excluding
                // the last element, which is the one that corresponds to the diagonal of the aggregated Hessian
                // matrix)...
                for (uint32 j = 0; j < i; j++) {
                    const DenseMappingVector<uint32>::Entry& bin2 = mappingIterator[j];
                    DenseMappingVector<uint32>::Entry::const_iterator end2 = bin2.cend();

                    // Again, ignore empty bins...
                    if (bin2.cbegin() != end2) {
                        float64 aggregatedHessians = 0;

                        // Iterate the label indices that belong to the bins at indices `currentBin` and `j`...
                        for (auto it = bin.cbegin(); it != end; it++) {
                            for (auto it2 = bin2.cbegin(); it2 != end2; it2++) {
                                uint32 labelIndex = *it;
                                uint32 labelIndex2 = *it2;
                                uint32 r, c;

                                if (labelIndex < labelIndex2) {
                                    r = labelIndex;
                                    c = labelIndex2;
                                } else {
                                    r = labelIndex2;
                                    c = labelIndex;
                                }

                                // Add the hessian at the `r`-th row and `c`-th column of the original Hessian matrix...
                                aggregatedHessians += hessianIterator[triangularNumber(c) + r];
                            }
                        }

                        // Copy the aggregated Hessians for the current row to the output array...
                        hessians[n] = aggregatedHessians;
                        n++;
                    }
                }

                // Copy the aggregated gradients, as well as the Hessians that belong to the diagonal of the Hessian
                // matrix, to the output arrays...
                gradients[binIndex] = aggregatedGradients;
                hessians[n] = aggregatedDiagonalHessians;
                n++;

                // Store the number of elements in the current bin...
                numElementsPerBin[binIndex] = numElements;
                binIndex++;
            }
        }

        return binIndex;
    }

    /**
     * Adds a specific L2 regularization weight to the diagonal of a coefficient matrix.
     *
     * @param output                    A pointer to an array of type `float64`, shape `(n, n)` that stores the
     *                                  coefficients
     * @param n                         The number of rows and columns in the coefficient matrix
     * @param numElementsPerBin         A pointer to an array of type `uint32`, shape `(n)` that stores the number of
     *                                  elements per bin
     * @param l2RegularizationWeight    The L2 regularization weight to be added
     */
    static inline void addRegularizationWeight(float64* output, uint32 n, const uint32* numElementsPerBin,
                                               float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < n; i++) {
            uint32 weight = numElementsPerBin[i];
            output[(i * n) + i] += (weight * l2RegularizationWeight);
        }
    }

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
     * Hessians that have been calculated according to a loss function that is applied example wise using L2
     * regularization. The labels are assigned to bins based on the corresponding gradients.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<class T>
    class BinningExampleWiseRuleEvaluation : public AbstractExampleWiseRuleEvaluation<T> {

        private:

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

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param maxBins                   The maximum number of bins to assign labels to
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
             *                                  different BLAS routines
             * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
             *                                  different LAPACK routines
             */
            BinningExampleWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight, uint32 maxBins,
                                             std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr,
                                             std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
                : AbstractExampleWiseRuleEvaluation<T>(labelIndices, lapackPtr),
                  l2RegularizationWeight_(l2RegularizationWeight), maxBins_(maxBins),
                  binningPtr_(std::move(binningPtr)), blasPtr_(blasPtr), scoreVector_(nullptr),
                  labelWiseScoreVector_(nullptr), tmpGradients_(nullptr), tmpHessians_(nullptr),
                  numElementsPerBin_(nullptr), mapping_(nullptr) {

            }

            ~BinningExampleWiseRuleEvaluation() {
                delete scoreVector_;
                delete labelWiseScoreVector_;
                free(tmpGradients_);
                free(tmpHessians_);
                free(numElementsPerBin_);
                delete mapping_;
            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseExampleWiseStatisticVector& statisticVector) override {
                if (labelWiseScoreVector_ == nullptr) {
                    labelWiseScoreVector_ = new DenseBinnedLabelWiseScoreVector<T>(this->labelIndices_, maxBins_ + 1);
                    tmpGradients_ = (float64*) malloc(maxBins_ * sizeof(float64));
                    tmpHessians_ = (float64*) malloc(maxBins_ * sizeof(float64));
                    numElementsPerBin_ = (uint32*) malloc(maxBins_ * sizeof(uint32));

                    // The last bin is used for labels with zero statistics. For this particular bin, the prediction and
                    // quality score is always zero.
                    labelWiseScoreVector_->scores_binned_begin()[maxBins_] = 0;
                    labelWiseScoreVector_->quality_scores_binned_begin()[maxBins_] = 0;
                }

                // Obtain information about the bins to be used...
                LabelInfo labelInfo = binningPtr_->getLabelInfo(statisticVector);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;
                labelWiseScoreVector_->setNumBins(numBins, false);

                // Reset arrays to zero...
                setArrayToZeros(tmpGradients_, numBins);
                setArrayToZeros(tmpHessians_, numBins);
                setArrayToZeros(numElementsPerBin_, numBins);

                // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
                auto callback = [this, &statisticVector](uint32 binIndex, uint32 labelIndex, float64 statistic) {
                    tmpGradients_[binIndex] += statistic;
                    float64 hessian = statisticVector.hessians_diagonal_cbegin()[labelIndex];
                    tmpHessians_[binIndex] += hessian;
                    numElementsPerBin_[binIndex] += 1;
                    labelWiseScoreVector_->indices_binned_begin()[labelIndex] = binIndex;
                };
                auto zeroCallback = [this](uint32 labelIndex) {
                    labelWiseScoreVector_->indices_binned_begin()[labelIndex] = maxBins_;
                };
                binningPtr_->createBins(labelInfo, statisticVector, callback, zeroCallback);

                // Compute predictions and quality scores...
                labelWiseScoreVector_->overallQualityScore = calculateLabelWisePredictionInternally<
                        typename DenseBinnedLabelWiseScoreVector<T>::score_binned_iterator,
                        typename DenseBinnedLabelWiseScoreVector<T>::quality_score_binned_iterator, float64*, float64*,
                        uint32*>(
                    numBins, labelWiseScoreVector_->scores_binned_begin(),
                    labelWiseScoreVector_->quality_scores_binned_begin(), tmpGradients_, tmpHessians_,
                    numElementsPerBin_, l2RegularizationWeight_);
                return *labelWiseScoreVector_;
            }

            const IScoreVector& calculateExampleWisePrediction(
                    DenseExampleWiseStatisticVector& statisticVector) override {
                if (scoreVector_ == nullptr) {
                    scoreVector_ = new DenseBinnedScoreVector<T>(this->labelIndices_, maxBins_ + 1);
                    this->initializeTmpArrays(maxBins_);
                    tmpGradients_ = (float64*) malloc(maxBins_ * sizeof(float64));
                    tmpHessians_ = (float64*) malloc(triangularNumber(maxBins_) * sizeof(float64));
                    numElementsPerBin_ = (uint32*) malloc(maxBins_ * sizeof(uint32));
                    mapping_ = new DenseMappingVector<uint32>(maxBins_);

                    // The last bin is used for labels with zero statistics. For this particular bin, the prediction is
                    // always zero.
                    scoreVector_->scores_binned_begin()[maxBins_] = 0;
                }

                // Obtain information about the bins to be used...
                LabelInfo labelInfo = binningPtr_->getLabelInfo(statisticVector);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;
                float64 qualityScore;

                if (numBins > 0) {
                    // Reset mapping...
                    mapping_->clear();

                    // Apply binning method in order to aggregate the gradients and Hessians that belong to the same
                    // bins...
                    auto callback = [this](uint32 binIndex, uint32 labelIndex, float64 statistic) {
                        mapping_->begin()[binIndex].push_front(labelIndex);
                    };
                    auto zeroCallback = [this](uint32 labelIndex) {
                        scoreVector_->indices_binned_begin()[labelIndex] = maxBins_;
                    };
                    binningPtr_->createBins(labelInfo, statisticVector, callback, zeroCallback);
                    numBins = aggregateGradientsAndHessians<T>(statisticVector, *mapping_, numElementsPerBin_,
                                                               *scoreVector_, tmpGradients_, tmpHessians_, numBins);
                    scoreVector_->setNumBins(numBins, false);

                    typename DenseBinnedScoreVector<T>::score_binned_iterator scoreIterator =
                        scoreVector_->scores_binned_begin();
                    copyCoefficients<float64*>(tmpHessians_, this->dsysvTmpArray1_, numBins);
                    addRegularizationWeight(this->dsysvTmpArray1_, numBins, numElementsPerBin_,
                                            l2RegularizationWeight_);
                    copyOrdinates<float64*>(tmpGradients_, scoreIterator, numBins);

                    // Calculate the scores to be predicted for the individual labels by solving a system of linear
                    // equations...
                    this->lapackPtr_->dsysv(this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_,
                                            scoreIterator, numBins, this->dsysvLwork_);

                    // Calculate the overall quality score...
                    qualityScore = calculateExampleWiseQualityScore(numBins, scoreIterator, tmpGradients_, tmpHessians_,
                                                                    *blasPtr_, this->dspmvTmpArray_);
                    qualityScore += 0.5 * l2RegularizationWeight_ *
                                    l2NormPow<typename DenseBinnedScoreVector<T>::score_binned_iterator, uint32*>(
                                        scoreIterator, numElementsPerBin_, numBins);
                } else {
                    setArrayToValue(scoreVector_->indices_binned_begin(), this->labelIndices_.getNumElements(),
                                    maxBins_);
                    qualityScore = 0;
                }

                scoreVector_->overallQualityScore = qualityScore;
                return *scoreVector_;
            }

    };

    EqualWidthBinningExampleWiseRuleEvaluationFactory::EqualWidthBinningExampleWiseRuleEvaluationFactory(
            float64 l2RegularizationWeight, float32 binRatio, uint32 minBins, uint32 maxBins,
            std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
        : l2RegularizationWeight_(l2RegularizationWeight), binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins),
          blasPtr_(blasPtr), lapackPtr_(lapackPtr) {

    }

    std::unique_ptr<IExampleWiseRuleEvaluation> EqualWidthBinningExampleWiseRuleEvaluationFactory::create(
            const FullIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr =
            std::make_unique<EqualWidthLabelBinning<DenseExampleWiseStatisticVector>>(binRatio_, minBins_, maxBins_);
        uint32 maxBins = binningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<BinningExampleWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                                   maxBins, std::move(binningPtr),
                                                                                   blasPtr_, lapackPtr_);
    }

    std::unique_ptr<IExampleWiseRuleEvaluation> EqualWidthBinningExampleWiseRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr =
            std::make_unique<EqualWidthLabelBinning<DenseExampleWiseStatisticVector>>(binRatio_, minBins_, maxBins_);
        uint32 maxBins = binningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<BinningExampleWiseRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                      l2RegularizationWeight_, maxBins,
                                                                                      std::move(binningPtr), blasPtr_,
                                                                                      lapackPtr_);
    }

}
