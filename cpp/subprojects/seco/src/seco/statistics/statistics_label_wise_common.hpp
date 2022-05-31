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
     * @tparam LabelMatrix              The type of the matrix that provides access to the labels of the training
     *                                  examples
     * @tparam CoverageMatrix           The type of the matrix that is used to store how often individual examples and
     *                                  labels have been covered
     * @tparam ConfusionMatrixVector    The type of the vector that is used to store confusion matrices
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     */
    template<typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector,
             typename RuleEvaluationFactory, typename WeightVector>
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
            class WeightedStatisticsSubset final : public IWeightedStatisticsSubset {

                private:

                    const LabelMatrix& labelMatrix_;

                    const CoverageMatrix& coverageMatrix_;

                    const BinarySparseArrayVector& majorityLabelVector_;

                    const ConfusionMatrixVector& totalSumVector_;

                    const ConfusionMatrixVector* subsetSumVector_;

                    const IndexVector& labelIndices_;

                    ConfusionMatrixVector sumVector_;

                    ConfusionMatrixVector tmpVector_;

                    std::unique_ptr<ConfusionMatrixVector> accumulatedSumVectorPtr_;

                    std::unique_ptr<ConfusionMatrixVector> totalCoverableSumVectorPtr_;

                    std::unique_ptr<IRuleEvaluation> ruleEvaluationPtr_;

                public:

                    /**
                     * @param statistics            A reference to an object of type `LabelWiseWeightedStatistics` that
                     *                              stores the confusion matrices
                     * @param ruleEvaluationFactory A reference to an object of type `RuleEvaluationFactory` that allows
                     *                              to create instances of the class that should be used for calculating
                     *                              the predictions of rules, as well as corresponding quality scores
                     * @param labelIndices          A reference to an object of template type `IndexVector` that
                     *                              provides access to the indices of the labels that are included in
                     *                              the subset
                     */
                    WeightedStatisticsSubset(const LabelWiseWeightedStatistics& statistics,
                                             const RuleEvaluationFactory& ruleEvaluationFactory,
                                             const IndexVector& labelIndices)
                        : labelMatrix_(statistics.labelMatrix_), coverageMatrix_(statistics.coverageMatrix_),
                          majorityLabelVector_(statistics.majorityLabelVector_),
                          totalSumVector_(statistics.totalSumVector_), subsetSumVector_(&statistics.subsetSumVector_),
                          labelIndices_(labelIndices),
                          sumVector_(ConfusionMatrixVector(labelIndices.getNumElements(), true)),
                          tmpVector_(ConfusionMatrixVector(labelIndices.getNumElements())),
                          ruleEvaluationPtr_(ruleEvaluationFactory.create(labelIndices)) {

                    }

                    /**
                     * @see `IWeightedStatisticsSubset::addToMissing`
                     */
                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        // Allocate a vector for storing the totals sums of confusion matrices, if necessary...
                        if (!totalCoverableSumVectorPtr_) {
                            totalCoverableSumVectorPtr_ = std::make_unique<ConfusionMatrixVector>(*subsetSumVector_);
                            subsetSumVector_ = totalCoverableSumVectorPtr_.get();
                        }

                        // For each label, subtract the confusion matrices of the example at the given index (weighted
                        // by the given weight) from the total sum of confusion matrices...
                        totalCoverableSumVectorPtr_->remove(statisticIndex, labelMatrix_, majorityLabelVector_,
                                                            coverageMatrix_, weight);
                    }

                    /**
                     * @see `IStatisticsSubset::addToSubset`
                     */
                    void addToSubset(uint32 statisticIndex, float64 weight) override {
                        sumVector_.addToSubset(statisticIndex, labelMatrix_, majorityLabelVector_, coverageMatrix_,
                                               labelIndices_, weight);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::resetSubset`
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
                        return ruleEvaluationPtr_->evaluate(majorityLabelVector_, totalSumVector_, sumVector_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::evaluateAccumulated`
                     */
                    const IScoreVector& evaluateAccumulated() override final {
                        return ruleEvaluationPtr_->evaluate(majorityLabelVector_, totalSumVector_,
                                                            *accumulatedSumVectorPtr_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::evaluateUncovered`
                     */
                    const IScoreVector& evaluateUncovered() override final {
                        tmpVector_.difference(subsetSumVector_->cbegin(), subsetSumVector_->cend(), labelIndices_,
                                              sumVector_.cbegin(), sumVector_.cend());
                        return ruleEvaluationPtr_->evaluate(majorityLabelVector_, totalSumVector_, tmpVector_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::evaluateUncoveredAccumulated`
                     */
                    const IScoreVector& evaluateUncoveredAccumulated() override final {
                        tmpVector_.difference(subsetSumVector_->cbegin(), subsetSumVector_->cend(), labelIndices_,
                                              accumulatedSumVectorPtr_->cbegin(), accumulatedSumVectorPtr_->cend());
                        return ruleEvaluationPtr_->evaluate(majorityLabelVector_, totalSumVector_, tmpVector_);
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
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param coverageMatrix        A reference to an object of template type `CoverageMatrix` that stores how
             *                              often individual examples and labels have been covered
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              predictions of the default rule
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions of rules, as well as corresponding quality scores
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             */
            LabelWiseWeightedStatistics(const LabelMatrix& labelMatrix, const CoverageMatrix& coverageMatrix,
                                        const BinarySparseArrayVector& majorityLabelVector,
                                        const RuleEvaluationFactory& ruleEvaluationFactory, const WeightVector& weights)
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
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
                    const CompleteIndexVector& labelIndices) const override final {
                return std::make_unique<WeightedStatisticsSubset<CompleteIndexVector>>(*this, ruleEvaluationFactory_,
                                                                                       labelIndices);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override final {
                return std::make_unique<WeightedStatisticsSubset<PartialIndexVector>>(*this, ruleEvaluationFactory_,
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
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param coverageMatrixPtr         An unique pointer to an object of template type `CoverageMatrix` that
             *                                  stores how often individual examples and labels have been covered
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `BinarySparseArrayVector` that
             *                                  stores the predictions of the default rule
             * @param ruleEvaluationFactory     A reference to an object of template type `RuleEvaluationFactory` that
             *                                  allows to create instances of the class that should be used for
             *                                  calculating the predictions of rules, as well as corresponding quality
             *                                  scores
             */
            AbstractLabelWiseStatistics(const LabelMatrix& labelMatrix,
                                        std::unique_ptr<CoverageMatrix> coverageMatrixPtr,
                                        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr,
                                        const RuleEvaluationFactory& ruleEvaluationFactory)
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
                return std::make_unique<LabelWiseWeightedStatistics<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                                    RuleEvaluationFactory, EqualWeightVector>>(
                    labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
                    const BitWeightVector& weights) const override final {
                return std::make_unique<LabelWiseWeightedStatistics<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                                    RuleEvaluationFactory, BitWeightVector>>(
                    labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
                    const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<LabelWiseWeightedStatistics<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                                    RuleEvaluationFactory, DenseWeightVector<uint32>>>(
                    labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights);
            }

    };

}
