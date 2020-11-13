#include "statistics_label_wise.h"
#include "data.h"
#include "data_label_wise.h"

using namespace boosting;


/**
 * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
 * applied label-wise.
 *
 * @tparam StatisticVector  The type of the vectors that are used to store gradients and Hessians
 * @tparam StatisticMatrix  The type of the matrices that are used to store gradients and Hessians
 * @tparam ScoreMatrix      The type of the matrices that are used to store predicted scores
 */
template<class StatisticVector, class StatisticMatrix, class ScoreMatrix>
class LabelWiseStatistics : public AbstractLabelWiseStatistics {

    private:

        /**
         * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
         * `LabelWiseStatistics`.
         *
         * @tparam T The type of the vector that provides access to the indices of the labels that are included in the
         *           subset
         */
        template<class T>
        class StatisticsSubset : public AbstractDecomposableStatisticsSubset {

            private:

                const LabelWiseStatistics& statistics_;

                std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr_;

                const T& labelIndices_;

                StatisticVector sumVector_;

                StatisticVector* accumulatedSumVector_;

                const StatisticVector* totalSumVector_;

                StatisticVector* totalCoverableSumVector_;

                StatisticVector* tmpVector_;

            public:

                /**
                 * @param statistics        A reference to an object of type `LabelWiseStatistics` that stores the
                 *                          gradients and Hessians
                 * @param ruleEvaluationPtr An unique pointer to an object of type `ILabelWiseRuleEvaluation` that
                 *                          should be used to calculate the predictions, as well as corresponding
                 *                          quality scores, of rules
                 * @param labelIndices      A reference to an object of template type `T` that provides access to the
                 *                          indices of the labels that are included in the subset
                 */
                StatisticsSubset(const LabelWiseStatistics& statistics,
                                 std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr, const T& labelIndices)
                    : statistics_(statistics), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)),
                      labelIndices_(labelIndices), sumVector_(StatisticVector(labelIndices.getNumElements(), true)),
                      totalSumVector_(&statistics_.totalSumVector_) {
                    accumulatedSumVector_ = nullptr;
                    totalCoverableSumVector_ = nullptr;
                    tmpVector_ = nullptr;
                }

                ~StatisticsSubset() {
                    delete accumulatedSumVector_;
                    delete totalCoverableSumVector_;
                    delete tmpVector_;
                }

                void addToMissing(uint32 statisticIndex, uint32 weight) override {
                    // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                    if (totalCoverableSumVector_ == nullptr) {
                        totalCoverableSumVector_ = new StatisticVector(*totalSumVector_);
                        totalSumVector_ = totalCoverableSumVector_;
                    }

                    // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                    // weight) from the total sums of gradients and Hessians...
                    totalCoverableSumVector_->subtract(
                        statistics_.statisticMatrix_->gradients_row_cbegin(statisticIndex),
                        statistics_.statisticMatrix_->gradients_row_cend(statisticIndex),
                        statistics_.statisticMatrix_->hessians_row_cbegin(statisticIndex),
                        statistics_.statisticMatrix_->hessians_row_cend(statisticIndex), weight);
                }

                void addToSubset(uint32 statisticIndex, uint32 weight) override {
                    sumVector_.addToSubset(statistics_.statisticMatrix_->gradients_row_cbegin(statisticIndex),
                                           statistics_.statisticMatrix_->gradients_row_cend(statisticIndex),
                                           statistics_.statisticMatrix_->hessians_row_cbegin(statisticIndex),
                                           statistics_.statisticMatrix_->hessians_row_cend(statisticIndex),
                                           labelIndices_, weight);
                }

                void resetSubset() override {
                    // Create a vector for storing the accumulated sums of gradients and Hessians, if necessary...
                    if (accumulatedSumVector_ == nullptr) {
                        uint32 numPredictions = labelIndices_.getNumElements();
                        accumulatedSumVector_ = new StatisticVector(numPredictions, true);
                    }

                    // Reset the sums of gradients and Hessians to zero and add it to the accumulated sums of gradients
                    // and Hessians...
                    accumulatedSumVector_->add(sumVector_.gradients_cbegin(), sumVector_.gradients_cend(),
                                               sumVector_.hessians_cbegin(), sumVector_.hessians_cend());
                    sumVector_.setAllToZero();
                }

                const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(bool uncovered,
                                                                                 bool accumulated) override {
                    const StatisticVector& sumsOfStatistics = accumulated ? *accumulatedSumVector_ : sumVector_;

                    if (uncovered) {
                        // Initialize temporary vector, if necessary...
                        if (tmpVector_ == nullptr) {
                            uint32 numPredictions = labelIndices_.getNumElements();
                            tmpVector_ = new StatisticVector(numPredictions);
                        }

                        tmpVector_->difference(totalSumVector_->gradients_cbegin(), totalSumVector_->gradients_cend(),
                                               totalSumVector_->hessians_cbegin(), totalSumVector_->hessians_cend(),
                                               labelIndices_, sumsOfStatistics.gradients_cbegin(),
                                               sumsOfStatistics.gradients_cend(), sumsOfStatistics.hessians_cbegin(),
                                               sumsOfStatistics.hessians_cend());
                        return ruleEvaluationPtr_->calculateLabelWisePrediction(*tmpVector_);
                    }

                    return ruleEvaluationPtr_->calculateLabelWisePrediction(sumsOfStatistics);
                }

        };

        /**
         * Allows to build a histogram based on the gradients and Hessians that are stored by an instance of the class
         * `LabelWiseStatistics`.
         */
        class HistogramBuilder : public IHistogramBuilder {

            private:

                const LabelWiseStatistics& statistics_;

                StatisticMatrix* statisticMatrix_;

            public:

                /**
                 * @param statistics    A reference to an object of type `LabelWiseStatistics` that stores the gradients
                 *                      and Hessians
                 * @param numBins       The number of bins, the histogram should consist of
                 */
                HistogramBuilder(const LabelWiseStatistics& statistics, uint32 numBins)
                    : statistics_(statistics),
                      statisticMatrix_(new StatisticMatrix(numBins, statistics.getNumLabels(), true)) {

                }

                void onBinUpdate(uint32 binIndex, const FeatureVector::Entry& entry) override {
                    uint32 index = entry.index;
                    statisticMatrix_->addToRow(binIndex, statistics_.statisticMatrix_->gradients_row_cbegin(index),
                                               statistics_.statisticMatrix_->gradients_row_cend(index),
                                               statistics_.statisticMatrix_->hessians_row_cbegin(index),
                                               statistics_.statisticMatrix_->hessians_row_cend(index));
                }

                std::unique_ptr<AbstractStatistics> build() const override {
                    return std::make_unique<LabelWiseStatistics<StatisticVector, StatisticMatrix, ScoreMatrix>>(
                        statistics_.lossFunctionPtr_, statistics_.ruleEvaluationFactoryPtr_,
                        statistics_.labelMatrixPtr_, statisticMatrix_, statistics_.scoreMatrix_);
                }

        };


        std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr_;

        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        StatisticMatrix* statisticMatrix_;

        ScoreMatrix* scoreMatrix_;

        StatisticVector totalSumVector_;

        template<class T>
        void applyPredictionInternally(uint32 statisticIndex, const T& prediction) {
            // Update the scores that are currently predicted for the example at the given index...
            scoreMatrix_->addToRowFromSubset(statisticIndex, prediction.scores_cbegin(), prediction.scores_cend(),
                                             prediction.indices_cbegin(), prediction.indices_cend());

            // Update the gradients and Hessians of the example at the given index...
            lossFunctionPtr_->updateStatistics(statisticIndex, *labelMatrixPtr_, *scoreMatrix_,
                                               prediction.indices_cbegin(), prediction.indices_cend(),
                                               *statisticMatrix_);
        }

    public:

        /**
         * @param lossFunctionPtr           A shared pointer to an object of type `AbstractLabelWiseLoss`, representing
         *                                  the loss function to be used for calculating gradients and Hessians
         * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`,
         *                                  that allows to create instances of the class that is used for calculating
         *                                  the predictions, as well as corresponding quality scores, of rules
         * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
         *                                  provides random access to the labels of the training examples
         * @param statisticMatrix           A pointer to an object of template type `StatisticMatrix` that stores the
         *                                  gradients and Hessians
         *                                  representing the Hessians
         * @param scoreMatrix               A pointer to an object of template type `ScoreMatrix` that stores the
         *                                  currently predicted scores
         */
        LabelWiseStatistics(std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr,
                            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr, StatisticMatrix* statisticMatrix,
                            ScoreMatrix* scoreMatrix)
            : AbstractLabelWiseStatistics(labelMatrixPtr->getNumExamples(), labelMatrixPtr->getNumLabels(),
                                          ruleEvaluationFactoryPtr),
              lossFunctionPtr_(lossFunctionPtr), labelMatrixPtr_(labelMatrixPtr), statisticMatrix_(statisticMatrix),
              scoreMatrix_(scoreMatrix), totalSumVector_(StatisticVector(labelMatrixPtr->getNumLabels())) {

        }

        ~LabelWiseStatistics() {
            delete statisticMatrix_;
            delete scoreMatrix_;
        }

        void resetCoveredStatistics() override {
            totalSumVector_.setAllToZero();
        }

        void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override {
            float64 signedWeight = remove ? -((float64) weight) : weight;
            totalSumVector_.add(statisticMatrix_->gradients_row_cbegin(statisticIndex),
                                statisticMatrix_->gradients_row_cend(statisticIndex),
                                statisticMatrix_->hessians_row_cbegin(statisticIndex),
                                statisticMatrix_->hessians_row_cend(statisticIndex), signedWeight);
        }

        std::unique_ptr<IStatisticsSubset> createSubset(const FullIndexVector& labelIndices) const override {
            std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr =
                ruleEvaluationFactoryPtr_->create(labelIndices);
            return std::make_unique<StatisticsSubset<FullIndexVector>>(*this, std::move(ruleEvaluationPtr),
                                                                       labelIndices);
        }

        std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices) const override {
            std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr =
                ruleEvaluationFactoryPtr_->create(labelIndices);
            return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, std::move(ruleEvaluationPtr),
                                                                          labelIndices);
        }

        void applyPrediction(uint32 statisticIndex, const FullPrediction& prediction) override {
            this->applyPredictionInternally<FullPrediction>(statisticIndex, prediction);
        }

        void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override {
            this->applyPredictionInternally<PartialPrediction>(statisticIndex, prediction);
        }

        std::unique_ptr<IHistogramBuilder> buildHistogram(uint32 numBins) const override {
            return std::make_unique<HistogramBuilder>(*this, numBins);
        }

};

AbstractLabelWiseStatistics::AbstractLabelWiseStatistics(
        uint32 numStatistics, uint32 numLabels,
        std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr)
    : AbstractGradientStatistics(numStatistics, numLabels), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr) {

}

void AbstractLabelWiseStatistics::setRuleEvaluationFactory(
        std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) {
    ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
}

DenseLabelWiseStatisticsFactoryImpl::DenseLabelWiseStatisticsFactoryImpl(
        std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr,
        std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr)
    : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
      labelMatrixPtr_(labelMatrixPtr) {

}

std::unique_ptr<AbstractLabelWiseStatistics> DenseLabelWiseStatisticsFactoryImpl::create() const {
    uint32 numExamples = labelMatrixPtr_->getNumExamples();
    uint32 numLabels = labelMatrixPtr_->getNumLabels();
    DenseLabelWiseStatisticMatrix* statisticMatrix = new DenseLabelWiseStatisticMatrix(numExamples, numLabels);
    DenseNumericMatrix<float64>* scoreMatrix = new DenseNumericMatrix<float64>(numExamples, numLabels, true);
    FullIndexVector labelIndices(numLabels);
    FullIndexVector::const_iterator labelIndicesBegin = labelIndices.cbegin();
    FullIndexVector::const_iterator labelIndicesEnd = labelIndices.cend();

    for (uint32 r = 0; r < numExamples; r++) {
        lossFunctionPtr_->updateStatistics(r, *labelMatrixPtr_, *scoreMatrix, labelIndicesBegin, labelIndicesEnd,
                                           *statisticMatrix);
    }

    return std::make_unique<LabelWiseStatistics<DenseLabelWiseStatisticVector, DenseLabelWiseStatisticMatrix, DenseNumericMatrix<float64>>>(
        lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrixPtr_, statisticMatrix, scoreMatrix);
}
