#include "seco/statistics/statistics_label_wise_dense.hpp"
#include "seco/data/matrix_dense_weights.hpp"
#include "seco/data/vector_dense_confusion_matrices.hpp"
#include "common/statistics/statistics_subset_decomposable.hpp"
#include "../data/confusion_matrices.hpp"


namespace seco {

    /**
     * Provides access to the elements of confusion matrices that are computed independently for each label using dense
     * data structures.
     */
    class LabelWiseStatistics final : public ILabelWiseStatistics {

        private:

            /**
             * Provides access to a subset of the confusion matrices that are stored by an instance of the class
             * `LabelWiseStatistics`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<class T>
            class StatisticsSubset final : public AbstractDecomposableStatisticsSubset {

                private:

                    const LabelWiseStatistics& statistics_;

                    const DenseConfusionMatrixVector* totalSumVector_;

                    std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr_;

                    const T& labelIndices_;

                    DenseConfusionMatrixVector sumVector_;

                    DenseConfusionMatrixVector* accumulatedSumVector_;

                    DenseConfusionMatrixVector* totalCoverableSumVector_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `LabelWiseStatistics` that stores the
                     *                          confusion matrices
                     * @param ruleEvaluationPtr An unique pointer to an object of type `ILabelWiseRuleEvaluation` that
                     *                          should be used to calculate the predictions, as well as corresponding
                     *                          quality scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubset(const LabelWiseStatistics& statistics,
                                     std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr, const T& labelIndices)
                        : statistics_(statistics), totalSumVector_(&statistics_.subsetSumVector_),
                          ruleEvaluationPtr_(std::move(ruleEvaluationPtr)), labelIndices_(labelIndices),
                          sumVector_(DenseConfusionMatrixVector(labelIndices.getNumElements(), true)),
                          accumulatedSumVector_(nullptr), totalCoverableSumVector_(nullptr) {

                    }

                    ~StatisticsSubset() {
                        delete accumulatedSumVector_;
                        delete totalCoverableSumVector_;
                    }

                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        // Allocate a vector for storing the totals sums of confusion matrices, if necessary...
                        if (totalCoverableSumVector_ == nullptr) {
                            totalCoverableSumVector_ = new DenseConfusionMatrixVector(*totalSumVector_);
                            totalSumVector_ = totalCoverableSumVector_;
                        }

                        // For each label, subtract the confusion matrices of the example at the given index (weighted
                        // by the given weight) from the total sum of confusion matrices...
                        totalCoverableSumVector_->add(statisticIndex, *statistics_.labelMatrixPtr_,
                                                      *statistics_.majorityLabelVectorPtr_,
                                                      *statistics_.weightMatrixPtr_, -weight);
                    }

                    void addToSubset(uint32 statisticIndex, float64 weight) override {
                        sumVector_.addToSubset(statisticIndex, *statistics_.labelMatrixPtr_,
                                               *statistics_.majorityLabelVectorPtr_, *statistics_.weightMatrixPtr_,
                                               labelIndices_.cbegin(), labelIndices_.cend(), weight);
                    }

                    void resetSubset() override {
                        // Allocate a vector for storing the accumulated confusion matrices, if necessary...
                        if (accumulatedSumVector_ == nullptr) {
                            uint32 numPredictions = labelIndices_.getNumElements();
                            accumulatedSumVector_ = new DenseConfusionMatrixVector(numPredictions, true);
                        }

                        // Reset the confusion matrix for each label to zero and add its elements to the accumulated
                        // confusion matrix...
                        accumulatedSumVector_->add(sumVector_.cbegin(), sumVector_.cend());
                        sumVector_.setAllToZero();
                    }

                    const ILabelWiseScoreVector& calculateLabelWisePrediction(bool uncovered,
                                                                              bool accumulated) override {
                        const DenseConfusionMatrixVector& sumsOfConfusionMatrices =
                            accumulated ? *accumulatedSumVector_ : sumVector_;
                        return ruleEvaluationPtr_->calculateLabelWisePrediction(*statistics_.majorityLabelVectorPtr_,
                                                                                statistics_.totalSumVector_,
                                                                                *totalSumVector_,
                                                                                sumsOfConfusionMatrices, uncovered);
                    }

            };

            uint32 numStatistics_;

            uint32 numLabels_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

            std::unique_ptr<DenseWeightMatrix> weightMatrixPtr_;

            // TODO Use sparse vector
            std::unique_ptr<DenseVector<uint8>> majorityLabelVectorPtr_;

            DenseConfusionMatrixVector totalSumVector_;

            DenseConfusionMatrixVector subsetSumVector_;

            template<class T>
            void applyPredictionInternally(uint32 statisticIndex, const T& prediction) {
                weightMatrixPtr_->updateRow(statisticIndex, *majorityLabelVectorPtr_, prediction.scores_cbegin(),
                    prediction.scores_cend(), prediction.indices_cbegin(), prediction.indices_cend());
            }

        public:

            /**
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             * @param weightMatrixPtr           An unique pointer to an object of type `DenseWeightMatrix` that stores
             *                                  the weights of individual examples and labels
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `DenseVector` that stores the
             *                                  predictions of the default rule
             */
            LabelWiseStatistics(std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                                std::unique_ptr<DenseWeightMatrix> weightMatrixPtr,
                                std::unique_ptr<DenseVector<uint8>> majorityLabelVectorPtr)
                : numStatistics_(labelMatrixPtr->getNumRows()), numLabels_(labelMatrixPtr->getNumCols()),
                  ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), labelMatrixPtr_(labelMatrixPtr),
                  weightMatrixPtr_(std::move(weightMatrixPtr)),
                  majorityLabelVectorPtr_(std::move(majorityLabelVectorPtr)),
                  totalSumVector_(DenseConfusionMatrixVector(numLabels_)),
                  subsetSumVector_(DenseConfusionMatrixVector(numLabels_)) {

            }

            uint32 getNumStatistics() const override {
                return numStatistics_;
            }

            uint32 getNumLabels() const override {
                return numLabels_;
            }

            float64 getSumOfUncoveredWeights() const override {
                return weightMatrixPtr_->getSumOfUncoveredWeights();
            }

            void setRuleEvaluationFactory(
                    std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) override {
                ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
            }

            void resetSampledStatistics() override {
                totalSumVector_.setAllToZero();
                subsetSumVector_.setAllToZero();
            }

            void addSampledStatistic(uint32 statisticIndex, float64 weight) override {
                uint32 numLabels = this->getNumLabels();
                typename DenseWeightMatrix::const_iterator weightIterator =
                    weightMatrixPtr_->row_cbegin(statisticIndex);
                DenseVector<uint8>::const_iterator majorityIterator = majorityLabelVectorPtr_->cbegin();

                for (uint32 c = 0; c < numLabels; c++) {
                    DenseConfusionMatrixVector::iterator totalIterator = totalSumVector_.confusion_matrix_begin(c);
                    DenseConfusionMatrixVector::iterator subsetIterator = subsetSumVector_.confusion_matrix_begin(c);
                    float64 labelWeight = weightIterator[c];

                    // Only uncovered labels must be considered...
                    if (labelWeight > 0) {
                        // Add the current example and label to the confusion matrix that corresponds to the current
                        // label...
                        uint8 trueLabel = labelMatrixPtr_->getValue(statisticIndex, c);
                        uint8 majorityLabel = majorityIterator[c];
                        uint32 element = getConfusionMatrixElement(trueLabel, majorityLabel);
                        totalIterator[element] += weight;
                        subsetIterator[element] += weight;
                    }
                }
            }

            void resetCoveredStatistics() override {
                subsetSumVector_.setAllToZero();
            }

            void updateCoveredStatistic(uint32 statisticIndex, float64 weight, bool remove) override {
                uint32 numLabels = this->getNumLabels();
                float64 signedWeight = remove ? -weight : weight;
                typename DenseWeightMatrix::const_iterator weightIterator =
                    weightMatrixPtr_->row_cbegin(statisticIndex);
                DenseVector<uint8>::const_iterator majorityIterator = majorityLabelVectorPtr_->cbegin();

                for (uint32 c = 0; c < numLabels; c++) {
                    DenseConfusionMatrixVector::iterator subsetIterator = subsetSumVector_.confusion_matrix_begin(c);
                    float64 labelWeight = weightIterator[c];

                    // Only uncovered labels must be considered...
                    if (labelWeight > 0) {
                        // Add the current example and label to the confusion matrix that corresponds to the current
                        // label...
                        uint8 trueLabel = labelMatrixPtr_->getValue(statisticIndex, c);
                        uint8 majorityLabel = majorityIterator[c];
                        uint32 element = getConfusionMatrixElement(trueLabel, majorityLabel);
                        subsetIterator[element] += signedWeight;
                    }
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

            float64 evaluatePrediction(uint32 statisticIndex, const IEvaluationMeasure& measure) const override {
                // TODO Support evaluation of predictions
                return 0;
            }

            std::unique_ptr<IHistogram> createHistogram(uint32 numBins) const override {
                //TODO Support creation of histograms
                return nullptr;
            }

    };

    DenseLabelWiseStatisticsFactory::DenseLabelWiseStatisticsFactory(
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr)
        : ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), labelMatrixPtr_(labelMatrixPtr) {

    }

    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory::create() const {
        uint32 numExamples = labelMatrixPtr_->getNumRows();
        uint32 numLabels = labelMatrixPtr_->getNumCols();
        std::unique_ptr<DenseWeightMatrix> weightMatrixPtr = std::make_unique<DenseWeightMatrix>(numExamples,
                                                                                                 numLabels);
        std::unique_ptr<DenseVector<uint8>> majorityLabelVectorPtr = std::make_unique<DenseVector<uint8>>(numLabels);
        DenseVector<uint8>::iterator majorityIterator = majorityLabelVectorPtr->begin();
        float64 threshold = numExamples / 2.0;
        float64 sumOfUncoveredWeights = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = 0;

            for (uint32 j = 0; j < numExamples; j++) {
                uint8 trueLabel = labelMatrixPtr_->getValue(j, i);
                numRelevant += trueLabel;
            }

            if (numRelevant > threshold) {
                majorityIterator[i] = 1;
                sumOfUncoveredWeights += (numExamples - numRelevant);
            } else {
                majorityIterator[i] = 0;
                sumOfUncoveredWeights += numRelevant;
            }
        }

        weightMatrixPtr->setSumOfUncoveredWeights(sumOfUncoveredWeights);
        return std::make_unique<LabelWiseStatistics>(ruleEvaluationFactoryPtr_, labelMatrixPtr_,
                                                     std::move(weightMatrixPtr), std::move(majorityLabelVectorPtr));
    }

}
