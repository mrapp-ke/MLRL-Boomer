/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/statistics/statistics_label_wise.hpp"
#include "common/data/vector_sparse_array_binary.hpp"


namespace seco {

    template<typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector>
    static inline void initializeLabelWiseSampledStatistics(const EqualWeightVector& weights,
                                                            const LabelMatrix& labelMatrix,
                                                            const BinarySparseArrayVector& majorityLabelVector,
                                                            const CoverageMatrix& coverageMatrix,
                                                            ConfusionMatrixVector& totalSumVector,
                                                            ConfusionMatrixVector& subsetSumVector) {
        uint32 numStatistics = weights.getNumElements();

        for (uint32 i = 0; i < numStatistics; i++) {
            totalSumVector.add(i, labelMatrix, majorityLabelVector, coverageMatrix, 1);
            subsetSumVector.add(i, labelMatrix, majorityLabelVector, coverageMatrix, 1);
        }
    }

    template<typename WeightVector, typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector>
    static inline void initializeLabelWiseSampledStatistics(const WeightVector& weights, const LabelMatrix& labelMatrix,
                                                            const BinarySparseArrayVector& majorityLabelVector,
                                                            const CoverageMatrix& coverageMatrix,
                                                            ConfusionMatrixVector& totalSumVector,
                                                            ConfusionMatrixVector& subsetSumVector) {
        uint32 numStatistics = weights.getNumElements();

        for (uint32 i = 0; i < numStatistics; i++) {
            float64 weight = weights.getWeight(i);
            totalSumVector.add(i, labelMatrix, majorityLabelVector, coverageMatrix, weight);
            subsetSumVector.add(i, labelMatrix, majorityLabelVector, coverageMatrix, weight);
        }
    }

    template<typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector>
    static inline void addLabelWiseStatistic(const EqualWeightVector& weights, const LabelMatrix& labelMatrix,
                                             const BinarySparseArrayVector& majorityLabelVector,
                                             const CoverageMatrix& coverageMatrix, ConfusionMatrixVector& vector,
                                             uint32 statisticIndex) {
        vector.add(statisticIndex, labelMatrix, majorityLabelVector, coverageMatrix, 1);
    }

    template<typename WeightVector, typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector>
    static inline void addLabelWiseStatistic(const WeightVector& weights, const LabelMatrix& labelMatrix,
                                             const BinarySparseArrayVector& majorityLabelVector,
                                             const CoverageMatrix& coverageMatrix, ConfusionMatrixVector& vector,
                                             uint32 statisticIndex) {
        float64 weight = weights.getWeight(statisticIndex);
        vector.add(statisticIndex, labelMatrix, majorityLabelVector, coverageMatrix, weight);
    }

    template<typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector>
    static inline void removeLabelWiseStatistic(const EqualWeightVector& weights, const LabelMatrix& labelMatrix,
                                                const BinarySparseArrayVector& majorityLabelVector,
                                                const CoverageMatrix& coverageMatrix, ConfusionMatrixVector& vector,
                                                uint32 statisticIndex) {
        vector.remove(statisticIndex, labelMatrix, majorityLabelVector, coverageMatrix, 1);
    }

    template<typename WeightVector, typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector>
    static inline void removeLabelWiseStatistic(const WeightVector& weights, const LabelMatrix& labelMatrix,
                                                const BinarySparseArrayVector& majorityLabelVector,
                                                const CoverageMatrix& coverageMatrix, ConfusionMatrixVector& vector,
                                                uint32 statisticIndex) {
        float64 weight = weights.getWeight(statisticIndex);
        vector.remove(statisticIndex, labelMatrix, majorityLabelVector, coverageMatrix, weight);
    }

    template<typename Prediction, typename CoverageMatrix>
    static inline void applyLabelWisePredictionInternally(uint32 statisticIndex, const Prediction& prediction,
                                                          CoverageMatrix& coverageMatrix,
                                                          const VectorConstView<uint32>& majorityLabelIndices) {
        coverageMatrix.increaseCoverage(statisticIndex, majorityLabelIndices, prediction.scores_cbegin(),
                                        prediction.scores_cend(), prediction.indices_cbegin(),
                                        prediction.indices_cend());
    }

    /**
     * An abstract base class for all statistics that provide access to the elements of weighted confusion matrices that
     * are computed independently for each label.
     *
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     * @tparam LabelMatrix              The type of the matrix that provides access to the labels of the training
     *                                  examples
     * @tparam CoverageMatrix           The type of the matrix that is used to store how often individual examples and
     *                                  labels have been covered
     * @tparam ConfusionMatrixVector    The type of the vector that is used to store confusion matrices
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename WeightVector, typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector,
             typename RuleEvaluationFactory>
    class LabelWiseWeightedStatistics : public IWeightedStatistics {

        private:

            /**
             * Provides access to a subset of the confusion matrices that are stored by an instance of the class
             * `LabelWiseWeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the labels that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class StatisticsSubset final : public IStatisticsSubset {

                private:

                    const LabelWiseWeightedStatistics& statistics_;

                    const ConfusionMatrixVector* subsetSumVector_;

                    std::unique_ptr<IRuleEvaluation> ruleEvaluationPtr_;

                    const IndexVector& labelIndices_;

                    ConfusionMatrixVector sumVector_;

                    ConfusionMatrixVector tmpVector_;

                    std::unique_ptr<ConfusionMatrixVector> accumulatedSumVectorPtr_;

                    std::unique_ptr<ConfusionMatrixVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `LabelWiseWeightedStatistics` that
                     *                          stores the confusion matrices
                     * @param ruleEvaluationPtr An unique pointer to an object of type `IRuleEvaluation` that should be
                     *                          used to calculate the predictions, as well as corresponding quality
                     *                          scores, of rules
                     * @param labelIndices      A reference to an object of template type `IndexVector` that provides
                     *                          access to the indices of the labels that are included in the subset
                     */
                    StatisticsSubset(const LabelWiseWeightedStatistics& statistics,
                                     std::unique_ptr<IRuleEvaluation> ruleEvaluationPtr,
                                     const IndexVector& labelIndices)
                        : statistics_(statistics), subsetSumVector_(&statistics_.subsetSumVector_),
                          ruleEvaluationPtr_(std::move(ruleEvaluationPtr)), labelIndices_(labelIndices),
                          sumVector_(ConfusionMatrixVector(labelIndices.getNumElements(), true)),
                          tmpVector_(ConfusionMatrixVector(labelIndices.getNumElements())) {

                    }

                    /**
                     * @see `IStatisticsSubset::addToMissing`
                     */
                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        // Allocate a vector for storing the totals sums of confusion matrices, if necessary...
                        if (!totalCoverableSumVectorPtr_) {
                            totalCoverableSumVectorPtr_ = std::make_unique<ConfusionMatrixVector>(*subsetSumVector_);
                            subsetSumVector_ = totalCoverableSumVectorPtr_.get();
                        }

                        // For each label, subtract the confusion matrices of the example at the given index (weighted
                        // by the given weight) from the total sum of confusion matrices...
                        totalCoverableSumVectorPtr_->remove(statisticIndex, statistics_.labelMatrix_,
                                                            statistics_.majorityLabelVector_,
                                                            statistics_.coverageMatrix_, weight);
                    }

                    /**
                     * @see `IStatisticsSubset::addToSubset`
                     */
                    void addToSubset(uint32 statisticIndex, float64 weight) override {
                        sumVector_.addToSubset(statisticIndex, statistics_.labelMatrix_,
                                               statistics_.majorityLabelVector_, statistics_.coverageMatrix_,
                                               labelIndices_, weight);
                    }

                    /**
                     * @see `IStatisticsSubset::resetSubset`
                     */
                    void resetSubset() override {
                        if (!accumulatedSumVectorPtr_) {
                            // Allocate a vector for storing the accumulated confusion matrices, if necessary...
                            accumulatedSumVectorPtr_ = std::make_unique<ConfusionMatrixVector>(sumVector_);
                        } else {
                            // Add the confusion matrix for each label to the accumulated confusion matrix...
                            accumulatedSumVectorPtr_->add(sumVector_.cbegin(), sumVector_.cend());
                        }

                        // Reset the confusion matrix for each label to zero...
                        sumVector_.clear();
                    }

                    /**
                     * @see `IStatisticsSubset::evaluate`
                     */
                    const IScoreVector& evaluate() override final {
                        return ruleEvaluationPtr_->evaluate(statistics_.majorityLabelVector_,
                                                            statistics_.totalSumVector_, sumVector_);
                    }

                    /**
                     * @see `IStatisticsSubset::evaluateAccumulated`
                     */
                    const IScoreVector& evaluateAccumulated() override final {
                        return ruleEvaluationPtr_->evaluate(statistics_.majorityLabelVector_,
                                                            statistics_.totalSumVector_, *accumulatedSumVectorPtr_);
                    }

                    /**
                     * @see `IStatisticsSubset::evaluateUncovered`
                     */
                    const IScoreVector& evaluateUncovered() override final {
                        tmpVector_.difference(subsetSumVector_->cbegin(), subsetSumVector_->cend(), labelIndices_,
                                              sumVector_.cbegin(), sumVector_.cend());
                        return ruleEvaluationPtr_->evaluate(statistics_.majorityLabelVector_,
                                                            statistics_.totalSumVector_, tmpVector_);
                    }

                    /**
                     * @see `IStatisticsSubset::evaluateUncoveredAccumulated`
                     */
                    const IScoreVector& evaluateUncoveredAccumulated() override final {
                        tmpVector_.difference(subsetSumVector_->cbegin(), subsetSumVector_->cend(), labelIndices_,
                                              accumulatedSumVectorPtr_->cbegin(), accumulatedSumVectorPtr_->cend());
                        return ruleEvaluationPtr_->evaluate(statistics_.majorityLabelVector_,
                                                            statistics_.totalSumVector_, tmpVector_);
                    }

            };

            const WeightVector& weights_;

            const RuleEvaluationFactory& ruleEvaluationFactory_;

            const LabelMatrix& labelMatrix_;

            const BinarySparseArrayVector& majorityLabelVector_;

            ConfusionMatrixVector totalSumVector_;

            ConfusionMatrixVector subsetSumVector_;

        protected:

            /**
             * A reference to an object of template type `CoverageMatrix` that stores how often individual examples and
             * labels have been covered.
             */
            const CoverageMatrix& coverageMatrix_;

        public:

            /**
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions of rules, as well as corresponding quality scores
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param coverageMatrix        A reference to an object of template type `CoverageMatrix` that stores how
             *                              often individual examples and labels have been covered
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              predictions of the default rule
             */
            LabelWiseWeightedStatistics(const WeightVector& weights, const RuleEvaluationFactory& ruleEvaluationFactory,
                                        const LabelMatrix& labelMatrix, const CoverageMatrix& coverageMatrix,
                                        const BinarySparseArrayVector& majorityLabelVector)
                : weights_(weights), ruleEvaluationFactory_(ruleEvaluationFactory), labelMatrix_(labelMatrix),
                  majorityLabelVector_(majorityLabelVector),
                  totalSumVector_(ConfusionMatrixVector(labelMatrix.getNumCols(), true)),
                  subsetSumVector_(ConfusionMatrixVector(labelMatrix.getNumCols(), true)),
                  coverageMatrix_(coverageMatrix) {
                initializeLabelWiseSampledStatistics(weights, labelMatrix, majorityLabelVector, coverageMatrix,
                                                     totalSumVector_, subsetSumVector_);
            }

            /**
             * @see `IImmutableWeightedStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return labelMatrix_.getNumRows();
            }

            /**
             * @see `IImmutableWeightedStatistics::getNumLabels`
             */
            uint32 getNumLabels() const override final {
                return labelMatrix_.getNumCols();
            }

            /**
             * @see `IWeightedStatistics::resetCoveredStatistics`
             */
            void resetCoveredStatistics() override final {
                subsetSumVector_.clear();
            }

            /**
             * @see `IWeightedStatistics::addCoveredStatistic`
             */
            void addCoveredStatistic(uint32 statisticIndex) override final {
                addLabelWiseStatistic(weights_, labelMatrix_, majorityLabelVector_, coverageMatrix_, subsetSumVector_,
                                      statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::removeCoveredStatistic`
             */
            void removeCoveredStatistic(uint32 statisticIndex) override final {
                removeLabelWiseStatistic(weights_, labelMatrix_, majorityLabelVector_, coverageMatrix_,
                                         subsetSumVector_, statisticIndex);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const CompleteIndexVector& labelIndices) const override final {
                std::unique_ptr<IRuleEvaluation> ruleEvaluationPtr = ruleEvaluationFactory_.create(labelIndices);
                return std::make_unique<StatisticsSubset<CompleteIndexVector>>(*this, std::move(ruleEvaluationPtr),
                                                                               labelIndices);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override final {
                std::unique_ptr<IRuleEvaluation> ruleEvaluationPtr = ruleEvaluationFactory_.create(labelIndices);
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, std::move(ruleEvaluationPtr),
                                                                              labelIndices);
            }

            /**
             * @see `IWeightedStatistics::createHistogram`
             */
            std::unique_ptr<IHistogram> createHistogram(uint32 numBins) const override final {
                //TODO Support creation of histograms
                return nullptr;
            }

    };

    /**
     * An abstract base class for all statistics that provide access to the elements of confusion matrices that are
     * computed independently for each label.
     *
     * @tparam LabelMatrix              The type of the matrix that provides access to the labels of the training
     *                                  examples
     * @tparam CoverageMatrix           The type of the matrix that is used to store how often individual examples and
     *                                  labels have been covered
     * @tparam ConfusionMatrixVector    The type of the vector that is used to store confusion matrices
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector,
             typename RuleEvaluationFactory>
    class AbstractLabelWiseStatistics : public ILabelWiseStatistics<RuleEvaluationFactory> {

        private:

            const RuleEvaluationFactory* ruleEvaluationFactory_;

            const LabelMatrix& labelMatrix_;

            std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr_;

            std::unique_ptr<CoverageMatrix> coverageMatrixPtr_;

        public:

            /**
             * @param ruleEvaluationFactory     A reference to an object of template type `RuleEvaluationFactory` that
             *                                  allows to create instances of the class that should be used for
             *                                  calculating the predictions of rules, as well as corresponding quality
             *                                  scores
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param coverageMatrixPtr         An unique pointer to an object of template type `CoverageMatrix` that
             *                                  stores how often individual examples and labels have been covered
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `BinarySparseArrayVector` that
             *                                  stores the predictions of the default rule
             */
            AbstractLabelWiseStatistics(const RuleEvaluationFactory& ruleEvaluationFactory,
                                        const LabelMatrix& labelMatrix,
                                        std::unique_ptr<CoverageMatrix> coverageMatrixPtr,
                                        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr)
                : ruleEvaluationFactory_(&ruleEvaluationFactory), labelMatrix_(labelMatrix),
                  majorityLabelVectorPtr_(std::move(majorityLabelVectorPtr)),
                  coverageMatrixPtr_(std::move(coverageMatrixPtr)) {

            }

            /**
             * @see `ICoverageStatistics::getSumOfUncoveredWeights`
             */
            float64 getSumOfUncoveredWeights() const override final {
                return coverageMatrixPtr_->getSumOfUncoveredWeights();
            }

            /**
             * @see `ILabelWiseStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(const RuleEvaluationFactory& ruleEvaluationFactory) override final {
                ruleEvaluationFactory_ = &ruleEvaluationFactory;
            }

            /**
             * @see `IStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return labelMatrix_.getNumRows();
            }

            /**
             * @see `IStatistics::getNumLabels`
             */
            uint32 getNumLabels() const override final {
                return labelMatrix_.getNumCols();
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const CompletePrediction& prediction) override final {
                applyLabelWisePredictionInternally<CompletePrediction, CoverageMatrix>(statisticIndex, prediction,
                                                                                       *coverageMatrixPtr_,
                                                                                       *majorityLabelVectorPtr_);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                applyLabelWisePredictionInternally<PartialPrediction, CoverageMatrix>(statisticIndex, prediction,
                                                                                      *coverageMatrixPtr_,
                                                                                      *majorityLabelVectorPtr_);
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                // TODO Support evaluation of predictions
                return 0;
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
                    const EqualWeightVector& weights) const override final {
                return std::make_unique<LabelWiseWeightedStatistics<EqualWeightVector, LabelMatrix, CoverageMatrix,
                                                                    ConfusionMatrixVector, RuleEvaluationFactory>>(
                    weights, *ruleEvaluationFactory_, labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
                    const BitWeightVector& weights) const override final {
                return std::make_unique<LabelWiseWeightedStatistics<BitWeightVector, LabelMatrix, CoverageMatrix,
                                                                    ConfusionMatrixVector, RuleEvaluationFactory>>(
                    weights, *ruleEvaluationFactory_, labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
                    const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<LabelWiseWeightedStatistics<DenseWeightVector<uint32>, LabelMatrix,
                                                                    CoverageMatrix, ConfusionMatrixVector,
                                                                    RuleEvaluationFactory>>(
                    weights, *ruleEvaluationFactory_, labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_);
            }

    };

}
