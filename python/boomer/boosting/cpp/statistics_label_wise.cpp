#include "statistics_label_wise.h"
#include "../../common/cpp/data_numeric.h"
#include "../../common/cpp/data_operations.h"
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

                float64* sumsOfGradients_;

                float64* accumulatedSumsOfGradients_;

                float64* sumsOfHessians_;

                float64* accumulatedSumsOfHessians_;

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
                      labelIndices_(labelIndices) {
                    uint32 numPredictions = labelIndices.getNumElements();
                    sumsOfGradients_ = (float64*) malloc(numPredictions * sizeof(float64));
                    setToZeros(sumsOfGradients_, numPredictions);
                    accumulatedSumsOfGradients_ = nullptr;
                    sumsOfHessians_ = (float64*) malloc(numPredictions * sizeof(float64));
                    setToZeros(sumsOfHessians_, numPredictions);
                    accumulatedSumsOfHessians_ = nullptr;
                }

                ~StatisticsSubset() {
                    free(sumsOfGradients_);
                    free(accumulatedSumsOfGradients_);
                    free(sumsOfHessians_);
                    free(accumulatedSumsOfHessians_);
                }

                void addToSubset(uint32 statisticIndex, uint32 weight) override {
                    // For each label, add the gradient and Hessian of the example at the given index (weighted by the
                    // given weight) to the current sum of gradients and Hessians...
                    uint32 offset = statisticIndex * statistics_.getNumLabels();
                    uint32 numPredictions = labelIndices_.getNumElements();
                    typename T::const_iterator indexIterator = labelIndices_.cbegin();

                    for (uint32 c = 0; c < numPredictions; c++) {
                        uint32 l = indexIterator[c];
                        uint32 i = offset + l;
                        sumsOfGradients_[c] += (weight * statistics_.gradients_[i]);
                        sumsOfHessians_[c] += (weight * statistics_.hessians_[i]);
                    }
                }

                void resetSubset() override {
                    uint32 numPredictions = labelIndices_.getNumElements();

                    // Allocate arrays for storing the accumulated sums of gradients and Hessians, if necessary...
                    if (accumulatedSumsOfGradients_ == nullptr) {
                        accumulatedSumsOfGradients_ = (float64*) malloc(numPredictions * sizeof(float64));
                        setToZeros(accumulatedSumsOfGradients_, numPredictions);
                        accumulatedSumsOfHessians_ = (float64*) malloc(numPredictions * sizeof(float64));
                        setToZeros(accumulatedSumsOfHessians_, numPredictions);
                    }

                    // Reset the sum of gradients and Hessians for each label to zero and add it to the accumulated sums
                    // of gradients and hessians...
                    for (uint32 c = 0; c < numPredictions; c++) {
                        accumulatedSumsOfGradients_[c] += sumsOfGradients_[c];
                        sumsOfGradients_[c] = 0;
                        accumulatedSumsOfHessians_[c] += sumsOfHessians_[c];
                        sumsOfHessians_[c] = 0;
                    }
                }

                const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(bool uncovered,
                                                                                 bool accumulated) override {
                    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
                    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;
                    return ruleEvaluationPtr_->calculateLabelWisePrediction(statistics_.totalSumsOfGradients_,
                                                                            sumsOfGradients,
                                                                            statistics_.totalSumsOfHessians_,
                                                                            sumsOfHessians, uncovered);
                }

        };

        /**
         * Allows to build a histogram based on the gradients and Hessians that are stored by an instance of the class
         * `DenseLabelWiseStatistics`.
         */
        class HistogramBuilder : public IHistogramBuilder {

            private:

                const DenseLabelWiseStatistics& statistics_;

                uint32 numBins_;

                DenseFloat64Matrix* gradientsM_;

                DenseFloat64Matrix* hessiansM_;

                // TODO Remove
                float64* gradients_;
                float64* hessians_;

            public:

                /**
                 * @param statistics    A reference to an object of type `DenseLabelWiseStatistics` that stores the
                 *                      gradients and Hessians
                 * @param numBins       The number of bins, the histogram should consist of
                 */
                HistogramBuilder(const DenseLabelWiseStatistics& statistics, uint32 numBins)
                    : statistics_(statistics), numBins_(numBins) {
                    uint32 numLabels = statistics.getNumLabels();
                    gradientsM_ = new DenseFloat64Matrix(numBins, numLabels, true);
                    hessiansM_ = new DenseFloat64Matrix(numBins, numLabels, true);

                    // TODO Remove
                    gradients_ = gradientsM_->begin();
                    hessians_ = hessiansM_->begin();
                }

                void onBinUpdate(uint32 binIndex, const FeatureVector::Entry& entry) override {
                    uint32 numLabels = statistics_.getNumLabels();
                    uint32 index = entry.index;
                    uint32 offset = index * numLabels;
                    uint32 binOffset = binIndex * numLabels;

                    for(uint32 c = 0; c < numLabels; c++) {
                        float64 gradient = statistics_.gradients_[offset + c];
                        float64 hessian = statistics_.hessians_[offset + c];
                        gradients_[binOffset + c] += gradient;
                        hessians_[binOffset + c] += hessian;
                    }
                }

                std::unique_ptr<AbstractStatistics> build() const override {
                    return std::make_unique<DenseLabelWiseStatistics>(statistics_.lossFunctionPtr_,
                                                                      statistics_.ruleEvaluationFactoryPtr_,
                                                                      statistics_.labelMatrixPtr_, gradientsM_,
                                                                      hessiansM_, statistics_.currentScoresM_);
                }

        };


        std::shared_ptr<ILabelWiseLoss> lossFunctionPtr_;

        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        DenseFloat64Matrix* gradientsM_;

        DenseFloat64Matrix* hessiansM_;

        DenseFloat64Matrix* currentScoresM_;

        DenseFloat64Vector totalSumsOfGradientsV_;

        DenseFloat64Vector totalSumsOfHessiansV_;

        // TODO Remove
        float64* gradients_;
        float64* hessians_;
        float64* currentScores_;
        float64* totalSumsOfGradients_;
        float64* totalSumsOfHessians_;

        template<class T>
        void applyPredictionInternally(uint32 statisticIndex, const T& prediction) {
            // Update the scores that are currently predicted for the example at the given index...
            add<float64>(prediction.scores_cbegin(), currentScoresM_->row_begin(statisticIndex),
                         prediction.indices_cbegin(), prediction.indices_cend());

            DenseFloat64Matrix::iterator gradientIterator = gradientsM_->row_begin(statisticIndex);
            DenseFloat64Matrix::iterator hessianIterator = hessiansM_->row_begin(statisticIndex);
            DenseFloat64Matrix::const_iterator scoreIterator = currentScoresM_->row_cbegin(statisticIndex);

            // Update the gradients and Hessians of the example at the given index...
            for (auto indexIterator = prediction.indices_cbegin(); indexIterator != prediction.indices_cend(); indexIterator++) {
                uint32 labelIndex = *indexIterator;
                float64 predictedScore = scoreIterator[labelIndex];
                std::pair<float64, float64> pair = lossFunctionPtr_->calculateGradientAndHessian(*labelMatrixPtr_,
                                                                                                 statisticIndex,
                                                                                                 labelIndex,
                                                                                                 predictedScore);
                gradientIterator[labelIndex] = pair.first;
                hessianIterator[labelIndex] = pair.second;
            }
        }

    public:

        /**
         * @param lossFunctionPtr           A shared pointer to an object of type `ILabelWiseLoss`, representing the
         *                                  loss function to be used for calculating gradients and Hessians
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
        DenseLabelWiseStatistics(std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
                                 std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                 std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                                 DenseFloat64Matrix* gradients, DenseFloat64Matrix* hessians,
                                 DenseFloat64Matrix* currentScores)
            : AbstractLabelWiseStatistics(labelMatrixPtr->getNumExamples(), labelMatrixPtr->getNumLabels(),
                                          ruleEvaluationFactoryPtr),
              lossFunctionPtr_(lossFunctionPtr), labelMatrixPtr_(labelMatrixPtr), gradientsM_(gradients),
              hessiansM_(hessians), currentScoresM_(currentScores),
              totalSumsOfGradientsV_(DenseFloat64Vector(labelMatrixPtr->getNumLabels())),
              totalSumsOfHessiansV_(DenseFloat64Vector(labelMatrixPtr->getNumLabels())) {
            // TODO Remove
            gradients_ = gradients->begin();
            hessians_ = hessians->begin();
            currentScores_ = currentScores->begin();
            totalSumsOfGradients_ = totalSumsOfGradientsV_.begin();
            totalSumsOfHessians_ = totalSumsOfHessiansV_.begin();
        }

        ~DenseLabelWiseStatistics() {
            delete gradientsM_;
            delete hessiansM_;
            delete currentScoresM_;
        }

        void resetCoveredStatistics() override {
            totalSumsOfGradientsV_.setAllToZero();
            totalSumsOfHessiansV_.setAllToZero();
        }

        void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override {
            uint32 numLabels = this->getNumLabels();
            uint32 offset = statisticIndex * numLabels;
            float64 signedWeight = remove ? -((float64) weight) : weight;

            // For each label, add the gradient and Hessian of the example at the given index (weighted by the given
            // weight) to the total sums of gradients and Hessians...
            for (uint32 c = 0; c < numLabels; c++) {
                uint32 i = offset + c;
                totalSumsOfGradients_[c] += (signedWeight * gradients_[i]);
                totalSumsOfHessians_[c] += (signedWeight * hessians_[i]);
            }
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
        std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
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

    for (uint32 r = 0; r < numExamples; r++) {
        DenseFloat64Matrix::iterator gradientIterator = gradients->row_begin(r);
        DenseFloat64Matrix::iterator hessianIterator = hessians->row_begin(r);

        for (uint32 c = 0; c < numLabels; c++) {
            // Calculate the initial gradient and Hessian for the current example and label...
            std::pair<float64, float64> pair = lossFunctionPtr_->calculateGradientAndHessian(*labelMatrixPtr_, r, c, 0);
            gradientIterator[c] = pair.first;
            hessianIterator[c] = pair.second;
        }
    }

    return std::make_unique<DenseLabelWiseStatistics>(lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrixPtr_,
                                                      gradients, hessians, currentScores);
}
