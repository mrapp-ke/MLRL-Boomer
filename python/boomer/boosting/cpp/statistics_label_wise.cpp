#include "statistics_label_wise.h"
#include "../../common/cpp/data_numeric.h"
#include <cstdlib>

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

                const DenseFloat64Vector* totalSumsOfGradients_;

                DenseFloat64Vector* totalSumsOfCoverableGradients_;

                const DenseFloat64Vector* totalSumsOfHessians_;

                DenseFloat64Vector* totalSumsOfCoverableHessians_;

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
                      totalSumsOfGradients_(&statistics_.totalSumsOfGradients_),
                      totalSumsOfHessians_(&statistics_.totalSumsOfHessians_) {
                    accumulatedSumsOfGradients_ = nullptr;
                    accumulatedSumsOfHessians_ = nullptr;
                    totalSumsOfCoverableGradients_ = nullptr;
                    totalSumsOfCoverableHessians_ = nullptr;
                    tmpGradients_ = nullptr;
                    tmpHessians_ = nullptr;
                }

                ~StatisticsSubset() {
                    delete accumulatedSumsOfGradients_;
                    delete accumulatedSumsOfHessians_;
                    delete totalSumsOfCoverableGradients_;
                    delete totalSumsOfCoverableHessians_;
                    delete tmpGradients_;
                    delete tmpHessians_;
                }

                void addToMissing(uint32 statisticIndex, uint32 weight) override {
                    // Create vectors for storing the totals sums of gradients and Hessians, if necessary...
                    if (totalSumsOfCoverableGradients_ == nullptr) {
                        totalSumsOfCoverableGradients_ = new DenseFloat64Vector(*totalSumsOfGradients_);
                        totalSumsOfCoverableHessians_ = new DenseFloat64Vector(*totalSumsOfHessians_);
                        totalSumsOfGradients_ = totalSumsOfCoverableGradients_;
                        totalSumsOfHessians_ = totalSumsOfCoverableHessians_;
                    }

                    // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                    // weight) from the total sum of gradients and Hessians...
                    totalSumsOfCoverableGradients_->subtract(statistics_.gradients_->row_cbegin(statisticIndex),
                                                             statistics_.gradients_->row_cend(statisticIndex), weight);
                    totalSumsOfCoverableHessians_->subtract(statistics_.hessians_->row_cbegin(statisticIndex),
                                                            statistics_.hessians_->row_cend(statisticIndex), weight);
                }

                void addToSubset(uint32 statisticIndex, uint32 weight) override {
                    sumsOfGradients_.addToSubset(statistics_.gradients_->row_cbegin(statisticIndex),
                                                 statistics_.gradients_->row_cend(statisticIndex), labelIndices_,
                                                 weight);
                    sumsOfHessians_.addToSubset(statistics_.hessians_->row_cbegin(statisticIndex),
                                                statistics_.hessians_->row_cend(statisticIndex), labelIndices_,
                                                weight);
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

                        typename T::const_iterator indexIterator = labelIndices_.cbegin();
                        DenseFloat64Vector::const_iterator totalSumsOfGradients = totalSumsOfGradients_->cbegin();
                        DenseFloat64Vector::const_iterator totalSumsOfHessians = totalSumsOfHessians_->cbegin();
                        DenseFloat64Vector::const_iterator sumsOfGradientsIterator = sumsOfGradients.cbegin();
                        DenseFloat64Vector::const_iterator sumsOfHessiansIterator = sumsOfHessians.cbegin();
                        DenseFloat64Vector::iterator tmpGradientsIterator = tmpGradients_->begin();
                        DenseFloat64Vector::iterator tmpHessiansIterator = tmpHessians_->begin();

                        for (uint32 c = 0; c < numPredictions; c++) {
                            uint32 l = indexIterator[c];
                            tmpGradientsIterator[c] = totalSumsOfGradients[l] - sumsOfGradientsIterator[c];
                            tmpHessiansIterator[c] = totalSumsOfHessians[l] - sumsOfHessiansIterator[c];
                        }

                        return ruleEvaluationPtr_->calculateLabelWisePrediction(tmpGradients_->cbegin(), tmpHessians_->cbegin());
                    }

                    return ruleEvaluationPtr_->calculateLabelWisePrediction(sumsOfGradients.cbegin(), sumsOfHessians.cbegin());
                }

        };

        /**
         * Allows to build a histogram based on the gradients and Hessians that are stored by an instance of the class
         * `DenseLabelWiseStatistics`.
         */
        class HistogramBuilder : public IHistogramBuilder {

            private:

                const DenseLabelWiseStatistics& statistics_;

                DenseFloat64Matrix* gradients_;

                DenseFloat64Matrix* hessians_;

            public:

                /**
                 * @param statistics    A reference to an object of type `DenseLabelWiseStatistics` that stores the
                 *                      gradients and Hessians
                 * @param numBins       The number of bins, the histogram should consist of
                 */
                HistogramBuilder(const DenseLabelWiseStatistics& statistics, uint32 numBins)
                    : statistics_(statistics), gradients_(new DenseFloat64Matrix(numBins, true)),
                      hessians_(new DenseFloat64Matrix(numBins, true)) {

                }

                void onBinUpdate(uint32 binIndex, const FeatureVector::Entry& entry) override {
                    uint32 index = entry.index;
                    gradients_->addToRow(binIndex, statistics_.gradients_->row_cbegin(index),
                                         statistics_.gradients_->row_cend(index));
                    hessians_->addToRow(binIndex, statistics_.hessians_->row_cbegin(index),
                                        statistics_.hessians_->row_cend(index));
                }

                std::unique_ptr<AbstractStatistics> build() const override {
                    return std::make_unique<DenseLabelWiseStatistics>(statistics_.lossFunctionPtr_,
                                                                      statistics_.ruleEvaluationFactoryPtr_,
                                                                      statistics_.labelMatrixPtr_, gradients_,
                                                                      hessians_, statistics_.currentScores_);
                }

        };


        std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr_;

        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        DenseFloat64Matrix* gradients_;

        DenseFloat64Matrix* hessians_;

        DenseFloat64Matrix* currentScores_;

        DenseFloat64Vector totalSumsOfGradients_;

        DenseFloat64Vector totalSumsOfHessians_;

        template<class T>
        void applyPredictionInternally(uint32 statisticIndex, const T& prediction) {
            // Update the scores that are currently predicted for the example at the given index...
            currentScores_->addToRowFromSubset(statisticIndex, prediction.scores_cbegin(), prediction.scores_cend(),
                                               prediction.indices_cbegin(), prediction.indices_cend());

            // Update the gradients and Hessians of the example at the given index...
            lossFunctionPtr_->updateGradientsAndHessians(*gradients_, *hessians_, *currentScores_, *labelMatrixPtr_,
                                                         statisticIndex, prediction.indices_cbegin(),
                                                         prediction.indices_cend());
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
         * @param gradients                 A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
         *                                  representing the gradients
         * @param hessians                  A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
         *                                  representing the Hessians
         * @param currentScores             A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
         *                                  representing the currently predicted scores
         */
        DenseLabelWiseStatistics(std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr,
                                 std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                 std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                                 DenseFloat64Matrix* gradients, DenseFloat64Matrix* hessians,
                                 DenseFloat64Matrix* currentScores)
            : AbstractLabelWiseStatistics(labelMatrixPtr->getNumExamples(), labelMatrixPtr->getNumLabels(),
                                          ruleEvaluationFactoryPtr),
              lossFunctionPtr_(lossFunctionPtr), labelMatrixPtr_(labelMatrixPtr), gradients_(gradients),
              hessians_(hessians), currentScores_(currentScores),
              totalSumsOfGradients_(DenseFloat64Vector(labelMatrixPtr->getNumLabels())),
              totalSumsOfHessians_(DenseFloat64Vector(labelMatrixPtr->getNumLabels())) {
        }

        ~DenseLabelWiseStatistics() {
            delete gradients_;
            delete hessians_;
            delete currentScores_;
        }

        void resetCoveredStatistics() override {
            totalSumsOfGradients_.setAllToZero();
            totalSumsOfHessians_.setAllToZero();
        }

        void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override {
            float64 signedWeight = remove ? -((float64) weight) : weight;
            totalSumsOfGradients_.add(gradients_->row_cbegin(statisticIndex), gradients_->row_cend(statisticIndex),
                                      signedWeight);
            totalSumsOfHessians_.add(hessians_->row_cbegin(statisticIndex), hessians_->row_cend(statisticIndex),
                                     signedWeight);
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
    DenseFloat64Matrix* gradients = new DenseFloat64Matrix(numExamples, numLabels);
    DenseFloat64Matrix* hessians = new DenseFloat64Matrix(numExamples, numLabels);
    DenseFloat64Matrix* currentScores = new DenseFloat64Matrix(numExamples, numLabels, true);
    FullIndexVector labelIndices(numLabels);
    FullIndexVector::const_iterator labelIndicesBegin = labelIndices.cbegin();
    FullIndexVector::const_iterator labelIndicesEnd = labelIndices.cend();

    for (uint32 r = 0; r < numExamples; r++) {
        lossFunctionPtr_->updateGradientsAndHessians(*gradients, *hessians, *currentScores, *labelMatrixPtr_, r,
                                                     labelIndicesBegin, labelIndicesEnd);
    }

    return std::make_unique<DenseLabelWiseStatistics>(lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrixPtr_,
                                                      gradients, hessians, currentScores);
}
