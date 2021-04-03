#include "seco/statistics/statistics_label_wise_dense.hpp"
#include "seco/data/matrix_dense_weights.hpp"
#include "seco/heuristics/confusion_matrices.hpp"
#include "common/data/arrays.hpp"
#include "common/statistics/statistics_subset_decomposable.hpp"
#include <cstdlib>


namespace seco {

    /**
     * Provides access to the elements of confusion matrices that are computed independently for each label using dense
     * data structures.
     *
     * @tparam WeightMatrix The type of the matrix that stores the weights of individual examples and labels
     */
    template<class WeightMatrix>
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

                    std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr_;

                    const T& labelIndices_;

                    float64* confusionMatricesCovered_;

                    float64* accumulatedConfusionMatricesCovered_;

                    float64* confusionMatricesSubset_;

                    float64* confusionMatricesCoverableSubset_;

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
                        : statistics_(statistics), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)),
                          labelIndices_(labelIndices) {
                        uint32 numPredictions = labelIndices.getNumElements();
                        confusionMatricesCovered_ =
                            (float64*) malloc(numPredictions * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
                        setArrayToZeros(confusionMatricesCovered_, numPredictions * NUM_CONFUSION_MATRIX_ELEMENTS);
                        accumulatedConfusionMatricesCovered_ = nullptr;
                        confusionMatricesSubset_ = statistics_.confusionMatricesSubset_;
                        confusionMatricesCoverableSubset_ = nullptr;
                    }

                    ~StatisticsSubset() {
                        free(confusionMatricesCovered_);
                        free(accumulatedConfusionMatricesCovered_);
                        free(confusionMatricesCoverableSubset_);
                    }

                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        uint32 numLabels = statistics_.getNumLabels();

                        // Allocate arrays for storing the totals sums of gradients and Hessians, if necessary...
                        if (confusionMatricesCoverableSubset_ == nullptr) {
                            confusionMatricesCoverableSubset_ = (float64*) malloc(numLabels * sizeof(float64));
                            copyArray(confusionMatricesSubset_, confusionMatricesCoverableSubset_, numLabels);
                            confusionMatricesSubset_ = confusionMatricesCoverableSubset_;
                        }

                        // For each label, subtract the gradient and Hessian of the example at the given index (weighted
                        // by the given weight) from the total sum of gradients and Hessians...
                        typename WeightMatrix::const_iterator weightIterator = statistics_.weightMatrixPtr_->row_cbegin(
                            statisticIndex);
                        DenseVector<uint8>::const_iterator majorityIterator =
                            statistics_.majorityLabelVectorPtr_->cbegin();

                        for (uint32 c = 0; c < numLabels; c++) {
                            uint8 labelWeight = weightIterator[c];

                            // Only uncovered labels must be considered...
                            if (labelWeight > 0) {
                                // Remove the current example and label from the confusion matrix that corresponds to
                                // the current label...
                                uint8 trueLabel = statistics_.labelMatrixPtr_->getValue(statisticIndex, c);
                                uint8 predictedLabel = majorityIterator[c] ? 0 : 1;
                                uint32 element = getConfusionMatrixElement(trueLabel, predictedLabel);
                                confusionMatricesSubset_[c * NUM_CONFUSION_MATRIX_ELEMENTS + element] -= weight;
                            }
                        }
                    }

                    void addToSubset(uint32 statisticIndex, float64 weight) override {
                        uint32 numPredictions = labelIndices_.getNumElements();
                        typename T::const_iterator indexIterator = labelIndices_.cbegin();
                        typename WeightMatrix::const_iterator weightIterator = statistics_.weightMatrixPtr_->row_cbegin(
                            statisticIndex);
                        DenseVector<uint8>::const_iterator majorityIterator =
                            statistics_.majorityLabelVectorPtr_->cbegin();

                        for (uint32 c = 0; c < numPredictions; c++) {
                            uint32 l = indexIterator[c];

                            // Only uncovered labels must be considered...
                            if (weightIterator[l] > 0) {
                                // Add the current example and label to the confusion matrix for the current label...
                                uint8 trueLabel = statistics_.labelMatrixPtr_->getValue(statisticIndex, l);
                                uint8 predictedLabel = majorityIterator[l] ? 0 : 1;
                                uint32 element = getConfusionMatrixElement(trueLabel, predictedLabel);
                                confusionMatricesCovered_[c * NUM_CONFUSION_MATRIX_ELEMENTS + element] += weight;
                            }
                        }
                    }

                    void resetSubset() override {
                        uint32 numPredictions = labelIndices_.getNumElements();

                        // Allocate an array for storing the accumulated confusion matrices, if necessary...
                        if (accumulatedConfusionMatricesCovered_ == nullptr) {
                            accumulatedConfusionMatricesCovered_ =
                                (float64*) malloc(numPredictions * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
                            setArrayToZeros(accumulatedConfusionMatricesCovered_,
                                            numPredictions * NUM_CONFUSION_MATRIX_ELEMENTS);
                        }

                        // Reset the confusion matrix for each label to zero and add its elements to the accumulated
                        // confusion matrix...
                        for (uint32 c = 0; c < numPredictions; c++) {
                            uint32 offset = c * NUM_CONFUSION_MATRIX_ELEMENTS;
                            copyArray(&confusionMatricesCovered_[offset], &accumulatedConfusionMatricesCovered_[offset],
                                      NUM_CONFUSION_MATRIX_ELEMENTS);
                            setArrayToZeros(&confusionMatricesCovered_[offset], NUM_CONFUSION_MATRIX_ELEMENTS);
                        }
                    }

                    const ILabelWiseScoreVector& calculateLabelWisePrediction(bool uncovered,
                                                                              bool accumulated) override {
                        float64* confusionMatricesCovered =
                            accumulated ? accumulatedConfusionMatricesCovered_ : confusionMatricesCovered_;
                        return ruleEvaluationPtr_->calculateLabelWisePrediction(*statistics_.majorityLabelVectorPtr_,
                                                                                statistics_.confusionMatricesTotal_,
                                                                                confusionMatricesSubset_,
                                                                                confusionMatricesCovered, uncovered);
                    }

            };

            uint32 numStatistics_;

            uint32 numLabels_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

            std::unique_ptr<WeightMatrix> weightMatrixPtr_;

            // TODO Use sparse vector
            std::unique_ptr<DenseVector<uint8>> majorityLabelVectorPtr_;

            float64* confusionMatricesTotal_;

            float64* confusionMatricesSubset_;

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
             * @param weightMatrixPtr           An unique pointer to an object of template type `WeightMatrix` that
             *                                  stores the weights of individual examples and labels
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `DenseVector` that stores the
             *                                  predictions of the default rule
             */
            LabelWiseStatistics(std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                                std::unique_ptr<WeightMatrix> weightMatrixPtr,
                                std::unique_ptr<DenseVector<uint8>> majorityLabelVectorPtr)
                : numStatistics_(labelMatrixPtr->getNumRows()), numLabels_(labelMatrixPtr->getNumCols()),
                  ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), labelMatrixPtr_(labelMatrixPtr),
                  weightMatrixPtr_(std::move(weightMatrixPtr)),
                  majorityLabelVectorPtr_(std::move(majorityLabelVectorPtr)) {
                // The number of labels
                uint32 numLabels = this->getNumLabels();
                // A matrix that stores a confusion matrix, which takes into account all examples, for each label
                confusionMatricesTotal_ =
                    (float64*) malloc(numLabels * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
                // A matrix that stores a confusion matrix, which takes into account the examples covered by the
                // previous refinement of a rule, for each label
                confusionMatricesSubset_ =
                    (float64*) malloc(numLabels * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
            }

            ~LabelWiseStatistics() {
                free(confusionMatricesTotal_);
                free(confusionMatricesSubset_);
            }

            uint32 getNumStatistics() const override {
                return numStatistics_;
            }

            uint32 getNumLabels() const override {
                return numLabels_;
            }

            float64 getSumOfUncoveredWeights() const override {
                return (float64) weightMatrixPtr_->getSumOfUncoveredWeights();
            }

            void setRuleEvaluationFactory(
                    std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) override {
                ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
            }

            void resetSampledStatistics() override {
                uint32 numLabels = this->getNumLabels();
                uint32 numElements = numLabels * NUM_CONFUSION_MATRIX_ELEMENTS;
                setArrayToZeros(confusionMatricesTotal_, numElements);
                setArrayToZeros(confusionMatricesSubset_, numElements);
            }

            void addSampledStatistic(uint32 statisticIndex, float64 weight) override {
                uint32 numLabels = this->getNumLabels();
                typename WeightMatrix::const_iterator weightIterator = weightMatrixPtr_->row_cbegin(statisticIndex);
                DenseVector<uint8>::const_iterator majorityIterator = majorityLabelVectorPtr_->cbegin();

                for (uint32 c = 0; c < numLabels; c++) {
                    uint8 labelWeight = weightIterator[c];

                    // Only uncovered labels must be considered...
                    if (labelWeight > 0) {
                        // Add the current example and label to the confusion matrix that corresponds to the current
                        // label...
                        uint8 trueLabel = labelMatrixPtr_->getValue(statisticIndex, c);
                        uint8 predictedLabel = majorityIterator[c] ? 0 : 1;
                        uint32 element = getConfusionMatrixElement(trueLabel, predictedLabel);
                        uint32 i = c * NUM_CONFUSION_MATRIX_ELEMENTS + element;
                        confusionMatricesTotal_[i] += weight;
                        confusionMatricesSubset_[i] += weight;
                    }
                }
            }

            void resetCoveredStatistics() override {
                // Reset confusion matrices to 0...
                uint32 numLabels = this->getNumLabels();
                uint32 numElements = numLabels * NUM_CONFUSION_MATRIX_ELEMENTS;
                setArrayToZeros(confusionMatricesSubset_, numElements);
            }

            void updateCoveredStatistic(uint32 statisticIndex, float64 weight, bool remove) override {
                uint32 numLabels = this->getNumLabels();
                float64 signedWeight = remove ? -weight : weight;
                typename WeightMatrix::const_iterator weightIterator = weightMatrixPtr_->row_cbegin(statisticIndex);
                DenseVector<uint8>::const_iterator majorityIterator = majorityLabelVectorPtr_->cbegin();

                for (uint32 c = 0; c < numLabels; c++) {
                    uint8 labelWeight = weightIterator[c];

                    // Only uncovered labels must be considered...
                    if (labelWeight > 0) {
                        // Add the current example and label to the confusion matrix that corresponds to the current
                        // label...
                        uint8 trueLabel = labelMatrixPtr_->getValue(statisticIndex, c);
                        uint8 predictedLabel = majorityIterator[c] ? 0 : 1;
                        uint32 element = getConfusionMatrixElement(trueLabel, predictedLabel);
                        confusionMatricesSubset_[c * NUM_CONFUSION_MATRIX_ELEMENTS + element] += signedWeight;
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
        uint32 sumOfUncoveredWeights = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = 0;

            for (uint32 j = 0; j < numExamples; j++) {
                uint8 trueLabel = labelMatrixPtr_->getValue(i, j);
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
        return std::make_unique<LabelWiseStatistics<DenseWeightMatrix>>(
            ruleEvaluationFactoryPtr_, labelMatrixPtr_, std::move(weightMatrixPtr), std::move(majorityLabelVectorPtr));
    }

}
