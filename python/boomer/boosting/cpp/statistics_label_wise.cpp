#include "statistics_label_wise.h"
#include "../../common/cpp/data_numeric.h"
#include "data_label_wise.h"

using namespace boosting;


/**
 * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
 * applied label-wise using dense data structures.
 */
class DenseLabelWiseStatistics : public AbstractLabelWiseStatistics {

    private:

        /**
         * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
         * `DenseLabelWiseStatistics`.
         *
         * @tparam T The type of the vector that provides access to the indices of the labels that are included in the
         *           subset
         */
        template<class T>
        class StatisticsSubset : public AbstractDecomposableStatisticsSubset {

            private:

                const DenseLabelWiseStatistics& statistics_;

                std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr_;

                const T& labelIndices_;

                DenseFloat64Vector sumsOfGradients_;

                DenseFloat64Vector sumsOfHessians_;

                DenseFloat64Vector* accumulatedSumsOfGradients_;

                DenseFloat64Vector* accumulatedSumsOfHessians_;

                const DenseLabelWiseStatisticsVector* totalSumsOfStatistics_;

                DenseLabelWiseStatisticsVector* totalSumsOfCoverableStatistics_;

                DenseFloat64Vector* tmpGradients_;

                DenseFloat64Vector* tmpHessians_;

            public:

                /**
                 * @param statistics        A reference to an object of type `DenseLabelWiseStatistics` that stores the
                 *                          gradients and Hessians
                 * @param ruleEvaluationPtr An unique pointer to an object of type `ILabelWiseRuleEvaluation` that
                 *                          should be used to calculate the predictions, as well as corresponding
                 *                          quality scores, of rules
                 * @param labelIndices      A reference to an object of template type `T` that provides access to the
                 *                          indices of the labels that are included in the subset
                 */
                StatisticsSubset(const DenseLabelWiseStatistics& statistics,
                                 std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr, const T& labelIndices)
                    : statistics_(statistics), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)),
                      labelIndices_(labelIndices),
                      sumsOfGradients_(DenseFloat64Vector(labelIndices.getNumElements(), true)),
                      sumsOfHessians_(DenseFloat64Vector(labelIndices.getNumElements(), true)),
                      totalSumsOfStatistics_(&statistics_.totalSumsOfStatistics_) {
                    accumulatedSumsOfGradients_ = nullptr;
                    accumulatedSumsOfHessians_ = nullptr;
                    totalSumsOfCoverableStatistics_ = nullptr;
                    tmpGradients_ = nullptr;
                    tmpHessians_ = nullptr;
                }

                ~StatisticsSubset() {
                    delete accumulatedSumsOfGradients_;
                    delete accumulatedSumsOfHessians_;
                    delete totalSumsOfCoverableStatistics_;
                    delete tmpGradients_;
                    delete tmpHessians_;
                }

                void addToMissing(uint32 statisticIndex, uint32 weight) override {
                    // Create vectors for storing the totals sums of gradients and Hessians, if necessary...
                    if (totalSumsOfCoverableStatistics_ == nullptr) {
                        totalSumsOfCoverableStatistics_ = new DenseLabelWiseStatisticsVector(*totalSumsOfStatistics_);
                        totalSumsOfStatistics_ = totalSumsOfCoverableStatistics_;
                    }

                    // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                    // weight) from the total sum of gradients and Hessians...
                    totalSumsOfCoverableStatistics_->subtract(
                        statistics_.statistics_->gradients_row_cbegin(statisticIndex),
                        statistics_.statistics_->gradients_row_cend(statisticIndex),
                        statistics_.statistics_->hessians_row_cbegin(statisticIndex),
                        statistics_.statistics_->hessians_row_cend(statisticIndex), weight);
                }

                void addToSubset(uint32 statisticIndex, uint32 weight) override {
                    sumsOfGradients_.addToSubset(statistics_.statistics_->gradients_row_cbegin(statisticIndex),
                                                 statistics_.statistics_->gradients_row_cend(statisticIndex),
                                                 labelIndices_, weight);
                    sumsOfHessians_.addToSubset(statistics_.statistics_->hessians_row_cbegin(statisticIndex),
                                                statistics_.statistics_->hessians_row_cend(statisticIndex),
                                                labelIndices_, weight);
                }

                void resetSubset() override {
                    uint32 numPredictions = labelIndices_.getNumElements();

                    // Allocate arrays for storing the accumulated sums of gradients and Hessians, if necessary...
                    if (accumulatedSumsOfGradients_ == nullptr) {
                        accumulatedSumsOfGradients_ = new DenseFloat64Vector(numPredictions, true);
                        accumulatedSumsOfHessians_ = new DenseFloat64Vector(numPredictions, true);
                    }

                    // Reset the sum of gradients and Hessians for each label to zero and add it to the accumulated sums
                    // of gradients and hessians...
                    accumulatedSumsOfGradients_->add(sumsOfGradients_.cbegin(), sumsOfGradients_.cend());
                    sumsOfGradients_.setAllToZero();
                    accumulatedSumsOfHessians_->add(sumsOfHessians_.cbegin(), sumsOfHessians_.cend());
                    sumsOfHessians_.setAllToZero();
                }

                const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(bool uncovered,
                                                                                 bool accumulated) override {
                    const DenseFloat64Vector& sumsOfGradients =
                        accumulated ? *accumulatedSumsOfGradients_ : sumsOfGradients_;
                    const DenseFloat64Vector& sumsOfHessians =
                        accumulated ? *accumulatedSumsOfHessians_ : sumsOfHessians_;

                    if (uncovered) {
                        uint32 numPredictions = labelIndices_.getNumElements();

                        // Initialize temporary vectors, if necessary...
                        if (tmpGradients_ == nullptr) {
                            tmpGradients_ = new DenseFloat64Vector(numPredictions);
                            tmpHessians_ = new DenseFloat64Vector(numPredictions);
                        }

                        tmpGradients_->difference(totalSumsOfStatistics_->gradients_cbegin(),
                                                  totalSumsOfStatistics_->gradients_cend(), labelIndices_,
                                                  sumsOfGradients.cbegin(), sumsOfGradients.cend());
                        tmpHessians_->difference(totalSumsOfStatistics_->hessians_cbegin(),
                                                 totalSumsOfStatistics_->hessians_cend(), labelIndices_,
                                                 sumsOfHessians.cbegin(), sumsOfHessians.cend());
                        return ruleEvaluationPtr_->calculateLabelWisePrediction(tmpGradients_->cbegin(),
                                                                                tmpGradients_->cend(),
                                                                                tmpHessians_->cbegin(),
                                                                                tmpHessians_->cend());
                    }

                    return ruleEvaluationPtr_->calculateLabelWisePrediction(sumsOfGradients.cbegin(),
                                                                            sumsOfGradients.cend(),
                                                                            sumsOfHessians.cbegin(),
                                                                            sumsOfHessians.cend());
                }

        };

        /**
         * Allows to build a histogram based on the gradients and Hessians that are stored by an instance of the class
         * `DenseLabelWiseStatistics`.
         */
        class HistogramBuilder : public IHistogramBuilder {

            private:

                const DenseLabelWiseStatistics& originalStatistics_;

                DenseLabelWiseStatisticsMatrix* statistics_;

            public:

                /**
                 * @param statistics    A reference to an object of type `DenseLabelWiseStatistics` that stores the
                 *                      gradients and Hessians
                 * @param numBins       The number of bins, the histogram should consist of
                 */
                HistogramBuilder(const DenseLabelWiseStatistics& statistics, uint32 numBins)
                    : originalStatistics_(statistics),
                      statistics_(new DenseLabelWiseStatisticsMatrix(numBins, statistics.getNumLabels(), true)) {

                }

                void onBinUpdate(uint32 binIndex, const FeatureVector::Entry& entry) override {
                    uint32 index = entry.index;
                    statistics_->addToRow(binIndex, originalStatistics_.statistics_->gradients_row_cbegin(index),
                                          originalStatistics_.statistics_->gradients_row_cend(index),
                                          originalStatistics_.statistics_->hessians_row_cbegin(index),
                                          originalStatistics_.statistics_->hessians_row_cend(index));
                }

                std::unique_ptr<AbstractStatistics> build() const override {
                    return std::make_unique<DenseLabelWiseStatistics>(originalStatistics_.lossFunctionPtr_,
                                                                      originalStatistics_.ruleEvaluationFactoryPtr_,
                                                                      originalStatistics_.labelMatrixPtr_, statistics_,
                                                                      originalStatistics_.currentScores_);
                }

        };


        std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr_;

        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        DenseLabelWiseStatisticsMatrix* statistics_;

        DenseFloat64Matrix* currentScores_;

        DenseLabelWiseStatisticsVector totalSumsOfStatistics_;

        template<class T>
        void applyPredictionInternally(uint32 statisticIndex, const T& prediction) {
            // Update the scores that are currently predicted for the example at the given index...
            currentScores_->addToRowFromSubset(statisticIndex, prediction.scores_cbegin(), prediction.scores_cend(),
                                               prediction.indices_cbegin(), prediction.indices_cend());

            // Update the gradients and Hessians of the example at the given index...
            lossFunctionPtr_->updateGradientsAndHessians(statisticIndex, *labelMatrixPtr_,
                                                         statistics_->gradients_row_begin(statisticIndex),
                                                         statistics_->gradients_row_end(statisticIndex),
                                                         statistics_->hessians_row_begin(statisticIndex),
                                                         statistics_->hessians_row_end(statisticIndex),
                                                         currentScores_->row_cbegin(statisticIndex),
                                                         currentScores_->row_cend(statisticIndex),
                                                         prediction.indices_cbegin(), prediction.indices_cend());
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
         * @param statistics                A pointer to an object of type `DenseLabelWiseStatisticsMatrix` that stores
         *                                  the gradients and Hessians
         *                                  representing the Hessians
         * @param currentScores             A pointer to an object of type `DenseFloat64Vector` that stores the
         *                                  currently predicted scores
         */
        DenseLabelWiseStatistics(std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr,
                                 std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                 std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                                 DenseLabelWiseStatisticsMatrix* statistics, DenseFloat64Matrix* currentScores)
            : AbstractLabelWiseStatistics(labelMatrixPtr->getNumExamples(), labelMatrixPtr->getNumLabels(),
                                          ruleEvaluationFactoryPtr),
              lossFunctionPtr_(lossFunctionPtr), labelMatrixPtr_(labelMatrixPtr), statistics_(statistics),
              currentScores_(currentScores),
              totalSumsOfStatistics_(DenseLabelWiseStatisticsVector(labelMatrixPtr->getNumLabels())) {

        }

        ~DenseLabelWiseStatistics() {
            delete statistics_;
            delete currentScores_;
        }

        void resetCoveredStatistics() override {
            totalSumsOfStatistics_.setAllToZero();
        }

        void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override {
            float64 signedWeight = remove ? -((float64) weight) : weight;
            totalSumsOfStatistics_.add(statistics_->gradients_row_cbegin(statisticIndex),
                                      statistics_->gradients_row_cend(statisticIndex),
                                      statistics_->hessians_row_cbegin(statisticIndex),
                                      statistics_->hessians_row_cend(statisticIndex), signedWeight);
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
    DenseLabelWiseStatisticsMatrix* statistics = new DenseLabelWiseStatisticsMatrix(numExamples, numLabels);
    DenseFloat64Matrix* currentScores = new DenseFloat64Matrix(numExamples, numLabels, true);
    FullIndexVector labelIndices(numLabels);
    FullIndexVector::const_iterator labelIndicesBegin = labelIndices.cbegin();
    FullIndexVector::const_iterator labelIndicesEnd = labelIndices.cend();

    for (uint32 r = 0; r < numExamples; r++) {
        lossFunctionPtr_->updateGradientsAndHessians(r, *labelMatrixPtr_, statistics->gradients_row_begin(r),
                                                     statistics->gradients_row_end(r),
                                                     statistics->hessians_row_begin(r), statistics->hessians_row_end(r),
                                                     currentScores->row_cbegin(r), currentScores->row_cend(r),
                                                     labelIndicesBegin, labelIndicesEnd);
    }

    return std::make_unique<DenseLabelWiseStatistics>(lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrixPtr_,
                                                      statistics, currentScores);
}
