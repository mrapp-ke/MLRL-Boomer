#include "statistics_label_wise.h"
#include "../../common/cpp/arrays.cpp"
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
                    uint32 numLabels = numBins_ * statistics.getNumLabels();
                    gradients_ = (float64*) calloc(numLabels, sizeof(float64));
                    hessians_ = (float64*) calloc(numLabels, sizeof(float64));
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
                                                                      statistics_.labelMatrixPtr_, gradients_,
                                                                      hessians_, statistics_.currentScores_);
                }

        };


        std::shared_ptr<ILabelWiseLoss> lossFunctionPtr_;

        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        float64* gradients_;

        float64* hessians_;

        float64* currentScores_;

        float64* totalSumsOfGradients_;

        float64* totalSumsOfHessians_;

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
                                 std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr, float64* gradients,
                                 float64* hessians, float64* currentScores)
            : AbstractLabelWiseStatistics(labelMatrixPtr->getNumExamples(), labelMatrixPtr->getNumLabels(),
              ruleEvaluationFactoryPtr), lossFunctionPtr_(lossFunctionPtr), labelMatrixPtr_(labelMatrixPtr),
              gradients_(gradients), hessians_(hessians), currentScores_(currentScores) {
            // The number of labels
            uint32 numLabels = this->getNumLabels();
            // An array that stores the column-wise sums of the matrix of gradients
            totalSumsOfGradients_ = (float64*) malloc(numLabels * sizeof(float64));
            // An array that stores the column-wise sums of the matrix of hessians
            totalSumsOfHessians_ = (float64*) malloc(numLabels * sizeof(float64));
        }

        ~DenseLabelWiseStatistics() {
            free(currentScores_);
            free(gradients_);
            free(totalSumsOfGradients_);
            free(hessians_);
            free(totalSumsOfHessians_);
        }

        void resetCoveredStatistics() override {
            uint32 numLabels = this->getNumLabels();
            setToZeros(totalSumsOfGradients_, numLabels);
            setToZeros(totalSumsOfHessians_, numLabels);
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
            uint32 numLabels = this->getNumLabels();
            uint32 offset = statisticIndex * numLabels;
            uint32 numPredictions = prediction.getNumElements();
            FullPrediction::score_const_iterator scoreIterator = prediction.scores_cbegin();

            // Only the labels that are predicted by the new rule must be considered...
            for (uint32 c = 0; c < numPredictions; c++) {
                // Update the score that is currently predicted for the current example and label...
                uint32 i = offset + c;
                float64 updatedScore = currentScores_[i] + scoreIterator[c];
                currentScores_[i] = updatedScore;

                // Update the gradient and Hessian for the current example and label...
                std::pair<float64, float64> pair = lossFunctionPtr_->calculateGradientAndHessian(*labelMatrixPtr_,
                                                                                                 statisticIndex, c,
                                                                                                 updatedScore);
                gradients_[i] = pair.first;
                hessians_[i] = pair.second;
            }
        }

        void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override {
            uint32 numLabels = this->getNumLabels();
            uint32 offset = statisticIndex * numLabels;
            uint32 numPredictions = prediction.getNumElements();
            PartialPrediction::score_const_iterator scoreIterator = prediction.scores_cbegin();
            PartialPrediction::index_const_iterator indexIterator = prediction.indices_cbegin();

            // Only the labels that are predicted by the new rule must be considered...
            for (uint32 c = 0; c < numPredictions; c++) {
                // Update the score that is currently predicted for the current example and label...
                uint32 l = indexIterator[c];
                uint32 i = offset + l;
                float64 updatedScore = currentScores_[i] + scoreIterator[c];
                currentScores_[i] = updatedScore;

                // Update the gradient and Hessian for the current example and label...
                std::pair<float64, float64> pair = lossFunctionPtr_->calculateGradientAndHessian(*labelMatrixPtr_,
                                                                                                 statisticIndex, l,
                                                                                                 updatedScore);
                gradients_[i] = pair.first;
                hessians_[i] = pair.second;
            }
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
    // The number of examples
    uint32 numExamples = labelMatrixPtr_->getNumExamples();
    // The number of labels
    uint32 numLabels = labelMatrixPtr_->getNumLabels();
    // A matrix that stores the gradients for each example and label
    float64* gradients = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the Hessians for each example and label
    float64* hessians = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the currently predicted scores for each example and label
    float64* currentScores = (float64*) malloc(numExamples * numLabels * sizeof(float64));

    for (uint32 r = 0; r < numExamples; r++) {
        uint32 offset = r * numLabels;

        for (uint32 c = 0; c < numLabels; c++) {
            uint32 i = offset + c;

            // Calculate the initial gradient and Hessian for the current example and label...
            std::pair<float64, float64> pair = lossFunctionPtr_->calculateGradientAndHessian(*labelMatrixPtr_, r, c, 0);
            gradients[i] = pair.first;
            hessians[i] = pair.second;

            // Store the score that is initially predicted for the current example and label...
            currentScores[i] = 0;
        }
    }

    return std::make_unique<DenseLabelWiseStatistics>(lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrixPtr_,
                                                      gradients, hessians, currentScores);
}
