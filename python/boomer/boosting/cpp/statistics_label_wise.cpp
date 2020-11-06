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

                DenseFloat64Vector sumsOfGradientsV_;

                DenseFloat64Vector sumsOfHessiansV_;

                DenseFloat64Vector* accumulatedSumsOfGradients_;

                DenseFloat64Vector* accumulatedSumsOfHessians_;

                // TODO Remove
                float64* sumsOfGradients_;
                float64* sumsOfHessians_;

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
                      sumsOfGradientsV_(DenseFloat64Vector(labelIndices.getNumElements(), true)),
                      sumsOfHessiansV_(DenseFloat64Vector(labelIndices.getNumElements(), true)) {
                    accumulatedSumsOfGradients_ = nullptr;
                    accumulatedSumsOfHessians_ = nullptr;

                    // TODO Remove
                    sumsOfGradients_ = sumsOfGradientsV_.begin();
                    sumsOfHessians_ = sumsOfHessiansV_.begin();
                }

                ~StatisticsSubset() {
                    delete accumulatedSumsOfGradients_;
                    delete accumulatedSumsOfHessians_;
                }

                void addToSubset(uint32 statisticIndex, uint32 weight) override {
                    vector::addToSubset<float64>(statistics_.gradientsM_->row_cbegin(statisticIndex),
                                                 sumsOfGradientsV_.begin(), labelIndices_.cbegin(),
                                                 labelIndices_.cend(), weight);
                    vector::addToSubset<float64>(statistics_.hessiansM_->row_cbegin(statisticIndex),
                                                 sumsOfHessiansV_.begin(), labelIndices_.cbegin(), labelIndices_.cend(),
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
                    vector::add<float64>(sumsOfGradientsV_.cbegin(), sumsOfGradientsV_.cend(),
                                         accumulatedSumsOfGradients_->begin());
                    sumsOfGradientsV_.setAllToZero();
                    vector::add<float64>(sumsOfHessiansV_.cbegin(), sumsOfHessiansV_.cend(),
                                         accumulatedSumsOfHessians_->begin());
                    sumsOfHessiansV_.setAllToZero();
                }

                const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(bool uncovered,
                                                                                 bool accumulated) override {
                    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_->begin() : sumsOfGradients_;
                    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_->begin() : sumsOfHessians_;
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
                    vector::add<float64>(statistics_.gradientsM_->row_cbegin(index),
                                         statistics_.gradientsM_->row_cend(index), gradients_->row_begin(binIndex));
                }

                std::unique_ptr<AbstractStatistics> build() const override {
                    return std::make_unique<DenseLabelWiseStatistics>(statistics_.lossFunctionPtr_,
                                                                      statistics_.ruleEvaluationFactoryPtr_,
                                                                      statistics_.labelMatrixPtr_, gradients_,
                                                                      hessians_, statistics_.currentScoresM_);
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
            vector::addFromSubset<float64>(prediction.scores_cbegin(), currentScoresM_->row_begin(statisticIndex),
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
            float64 signedWeight = remove ? -((float64) weight) : weight;
            vector::add<float64>(gradientsM_->row_cbegin(statisticIndex), gradientsM_->row_cend(statisticIndex),
                                 totalSumsOfGradientsV_.begin(), signedWeight);
            vector::add<float64>(hessiansM_->row_cbegin(statisticIndex), hessiansM_->row_cend(statisticIndex),
                                 totalSumsOfHessiansV_.begin(), signedWeight);
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
