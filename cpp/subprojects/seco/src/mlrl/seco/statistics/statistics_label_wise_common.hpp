/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/vector_sparse_array_binary.hpp"
#include "mlrl/seco/statistics/statistics_label_wise.hpp"

namespace seco {

    static inline bool hasNonZeroWeightLabelWise(const EqualWeightVector& weights, uint32 statisticIndex) {
        return true;
    }

    template<typename WeightVector>
    static inline bool hasNonZeroWeightLabelWise(const WeightVector& weights, uint32 statisticIndex) {
        return weights[statisticIndex] != 0;
    }

    template<typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector, typename IndexVector>
    static inline void addLabelWiseStatisticToSubset(const EqualWeightVector& weights, const LabelMatrix& labelMatrix,
                                                     const BinarySparseArrayVector& majorityLabelVector,
                                                     const CoverageMatrix& coverageMatrix,
                                                     ConfusionMatrixVector& vector, const IndexVector& labelIndices,
                                                     uint32 statisticIndex) {
        vector.addToSubset(statisticIndex, labelMatrix, majorityLabelVector, coverageMatrix, labelIndices, 1);
    }

    template<typename WeightVector, typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector,
             typename IndexVector>
    static inline void addLabelWiseStatisticToSubset(const WeightVector& weights, const LabelMatrix& labelMatrix,
                                                     const BinarySparseArrayVector& majorityLabelVector,
                                                     const CoverageMatrix& coverageMatrix,
                                                     ConfusionMatrixVector& vector, const IndexVector& labelIndices,
                                                     uint32 statisticIndex) {
        float64 weight = weights[statisticIndex];
        vector.addToSubset(statisticIndex, labelMatrix, majorityLabelVector, coverageMatrix, labelIndices, weight);
    }

    /**
     * An abstract base class for all subsets of confusion matrices that are computed independently for each label.
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
     * @tparam IndexVector              The type of the vector that provides access to the indices of the labels that
     *                                  are included in the subset
     */
    template<typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector,
             typename RuleEvaluationFactory, typename WeightVector, typename IndexVector>
    class AbstractLabelWiseStatisticsSubset : virtual public IStatisticsSubset {
        protected:

            /**
             * An object of type `ConfusionMatrixVector` that stores the sums of confusion matrix elements.
             */
            ConfusionMatrixVector sumVector_;

            /**
             * A reference to an object of template type `LabelMatrix` that provides access to the labels of the
             * training examples.
             */
            const LabelMatrix& labelMatrix_;

            /**
             * A reference to an object of template type `CoverageMatrix` that stores how often individual examples and
             * labels have been covered.
             */
            const CoverageMatrix& coverageMatrix_;

            /**
             * A reference to an object of type `BinarySparseArrayVector` that stores the predictions of the default
             * rule.
             */
            const BinarySparseArrayVector& majorityLabelVector_;

            /**
             * A reference to an object of template type `ConfusionMatrixVector` that stores the total sums of confusion
             * matrix elements.
             */
            const ConfusionMatrixVector& totalSumVector_;

            /**
             * A reference to an object of template type `WeightVector` that provides access to the weights of
             * individual statistics.
             */
            const WeightVector& weights_;

            /**
             * A reference to an object of template type `IndexVector` that provides access to the indices of the labels
             * that are included in the subset.
             */
            const IndexVector& labelIndices_;

            /**
             * An unique pointer to an object of type `IRuleEvaluation` that is used for calculating the predictions of
             * rules, as well as their overall quality.
             */
            const std::unique_ptr<IRuleEvaluation> ruleEvaluationPtr_;

        public:

            /**
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param coverageMatrix        A reference to an object of template type `CoverageMatrix` that stores how
             *                              often individual examples and labels have been covered
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              predictions of the default rule
             * @param totalSumVector        A reference to an object of template type `ConfusionMatrixVector` that
             *                              stores the total sums of confusion matrix elements
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param labelIndices          A reference to an object of template type `IndexVector` that
             *                              provides access to the indices of the labels that are included in
             *                              the subset
             */
            AbstractLabelWiseStatisticsSubset(const LabelMatrix& labelMatrix, const CoverageMatrix& coverageMatrix,
                                              const BinarySparseArrayVector& majorityLabelVector,
                                              const ConfusionMatrixVector& totalSumVector,
                                              const RuleEvaluationFactory& ruleEvaluationFactory,
                                              const WeightVector& weights, const IndexVector& labelIndices)
                : sumVector_(ConfusionMatrixVector(labelIndices.getNumElements(), true)), labelMatrix_(labelMatrix),
                  coverageMatrix_(coverageMatrix), majorityLabelVector_(majorityLabelVector),
                  totalSumVector_(totalSumVector), weights_(weights), labelIndices_(labelIndices),
                  ruleEvaluationPtr_(ruleEvaluationFactory.create(labelIndices)) {}

            /**
             * @see `IStatisticsSubset::hasNonZeroWeight`
             */
            bool hasNonZeroWeight(uint32 statisticIndex) const override final {
                return hasNonZeroWeightLabelWise(weights_, statisticIndex);
            }

            /**
             * @see `IStatisticsSubset::addToSubset`
             */
            void addToSubset(uint32 statisticIndex) override final {
                addLabelWiseStatisticToSubset(weights_, labelMatrix_, majorityLabelVector_, coverageMatrix_, sumVector_,
                                              labelIndices_, statisticIndex);
            }

            /**
             * @see `IStatisticsSubset::calculateScores`
             */
            const IScoreVector& calculateScores() override final {
                return ruleEvaluationPtr_->calculateScores(majorityLabelVector_, totalSumVector_, sumVector_);
            }
    };

    template<typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector>
    static inline void initializeLabelWiseStatisticVector(const EqualWeightVector& weights,
                                                          const LabelMatrix& labelMatrix,
                                                          const BinarySparseArrayVector& majorityLabelVector,
                                                          const CoverageMatrix& coverageMatrix,
                                                          ConfusionMatrixVector& statisticVector) {
        uint32 numStatistics = weights.getNumElements();

        for (uint32 i = 0; i < numStatistics; i++) {
            statisticVector.add(i, labelMatrix, majorityLabelVector, coverageMatrix, 1);
        }
    }

    template<typename WeightVector, typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector>
    static inline void initializeLabelWiseStatisticVector(const WeightVector& weights, const LabelMatrix& labelMatrix,
                                                          const BinarySparseArrayVector& majorityLabelVector,
                                                          const CoverageMatrix& coverageMatrix,
                                                          ConfusionMatrixVector& statisticVector) {
        uint32 numStatistics = weights.getNumElements();

        for (uint32 i = 0; i < numStatistics; i++) {
            float64 weight = weights[i];
            statisticVector.add(i, labelMatrix, majorityLabelVector, coverageMatrix, weight);
        }
    }

    /**
     * A subset of confusion matrices that are computed independently for each label.
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
     * @tparam IndexVector              The type of the vector that provides access to the indices of the labels that
     *                                  are included in the subset
     */
    template<typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector,
             typename RuleEvaluationFactory, typename WeightVector, typename IndexVector>
    class LabelWiseStatisticsSubset final
        : public AbstractLabelWiseStatisticsSubset<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                   RuleEvaluationFactory, WeightVector, IndexVector> {
        private:

            const std::unique_ptr<ConfusionMatrixVector> totalSumVectorPtr_;

        public:

            /**
             * @param totalSumVectorPtr     An unique pointer to an object of template type `ConfusionMatrixVector` that
             *                              stores the total sums of confusion matrix elements
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param coverageMatrix        A reference to an object of template type `CoverageMatrix` that stores how
             *                              often individual examples and labels have been covered
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              predictions of the default rule
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param labelIndices          A reference to an object of template type `IndexVector` that
             *                              provides access to the indices of the labels that are included in
             *                              the subset
             */
            LabelWiseStatisticsSubset(std::unique_ptr<ConfusionMatrixVector> totalSumVectorPtr,
                                      const LabelMatrix& labelMatrix, const CoverageMatrix& coverageMatrix,
                                      const BinarySparseArrayVector& majorityLabelVector,
                                      const RuleEvaluationFactory& ruleEvaluationFactory, const WeightVector& weights,
                                      const IndexVector& labelIndices)
                : AbstractLabelWiseStatisticsSubset<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                    RuleEvaluationFactory, WeightVector, IndexVector>(
                  labelMatrix, coverageMatrix, majorityLabelVector, *totalSumVectorPtr, ruleEvaluationFactory, weights,
                  labelIndices),
                  totalSumVectorPtr_(std::move(totalSumVectorPtr)) {
                initializeLabelWiseStatisticVector(weights, labelMatrix, majorityLabelVector, coverageMatrix,
                                                   *totalSumVectorPtr_);
            }
    };

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
        float64 weight = weights[statisticIndex];
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
        float64 weight = weights[statisticIndex];
        vector.remove(statisticIndex, labelMatrix, majorityLabelVector, coverageMatrix, weight);
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
    class LabelWiseWeightedStatistics final : public IWeightedStatistics {
        private:

            /**
             * Provides access to a subset of the confusion matrices that are stored by an instance of the class
             * `LabelWiseWeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the labels that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class WeightedStatisticsSubset final
                : virtual public IWeightedStatisticsSubset,
                  public AbstractLabelWiseStatisticsSubset<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                           RuleEvaluationFactory, WeightVector, IndexVector> {
                private:

                    const ConfusionMatrixVector* subsetSumVector_;

                    ConfusionMatrixVector tmpVector_;

                    std::unique_ptr<ConfusionMatrixVector> accumulatedSumVectorPtr_;

                    std::unique_ptr<ConfusionMatrixVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param statistics            A reference to an object of type `LabelWiseWeightedStatistics` that
                     *                              stores the confusion matrices
                     * @param labelIndices          A reference to an object of template type `IndexVector` that
                     *                              provides access to the indices of the labels that are included in
                     *                              the subset
                     */
                    WeightedStatisticsSubset(const LabelWiseWeightedStatistics& statistics,
                                             const IndexVector& labelIndices)
                        : AbstractLabelWiseStatisticsSubset<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                            RuleEvaluationFactory, WeightVector, IndexVector>(
                          statistics.labelMatrix_, statistics.coverageMatrix_, statistics.majorityLabelVector_,
                          statistics.totalSumVector_, statistics.ruleEvaluationFactory_, statistics.weights_,
                          labelIndices),
                          subsetSumVector_(&statistics.subsetSumVector_),
                          tmpVector_(ConfusionMatrixVector(labelIndices.getNumElements())) {}

                    /**
                     * @see `IWeightedStatisticsSubset::addToMissing`
                     */
                    void addToMissing(uint32 statisticIndex) override {
                        // Allocate a vector for storing the totals sums of confusion matrices, if necessary...
                        if (!totalCoverableSumVectorPtr_) {
                            totalCoverableSumVectorPtr_ = std::make_unique<ConfusionMatrixVector>(*subsetSumVector_);
                            subsetSumVector_ = totalCoverableSumVectorPtr_.get();
                        }

                        // For each label, subtract the confusion matrices of the example at the given index (weighted
                        // by the given weight) from the total sum of confusion matrices...

                        removeLabelWiseStatistic(this->weights_, this->labelMatrix_, this->majorityLabelVector_,
                                                 this->coverageMatrix_, *totalCoverableSumVectorPtr_, statisticIndex);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::resetSubset`
                     */
                    void resetSubset() override {
                        if (!accumulatedSumVectorPtr_) {
                            // Allocate a vector for storing the accumulated confusion matrices, if necessary...
                            accumulatedSumVectorPtr_ = std::make_unique<ConfusionMatrixVector>(this->sumVector_);
                        } else {
                            // Add the confusion matrix for each label to the accumulated confusion matrix...
                            accumulatedSumVectorPtr_->add(this->sumVector_.cbegin(), this->sumVector_.cend());
                        }

                        // Reset the confusion matrix for each label to zero...
                        this->sumVector_.clear();
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::calculateScoresAccumulated`
                     */
                    const IScoreVector& calculateScoresAccumulated() override {
                        return this->ruleEvaluationPtr_->calculateScores(
                          this->majorityLabelVector_, this->totalSumVector_, *accumulatedSumVectorPtr_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::calculateScoresUncovered`
                     */
                    const IScoreVector& calculateScoresUncovered() override {
                        tmpVector_.difference(subsetSumVector_->cbegin(), subsetSumVector_->cend(), this->labelIndices_,
                                              this->sumVector_.cbegin(), this->sumVector_.cend());
                        return this->ruleEvaluationPtr_->calculateScores(this->majorityLabelVector_,
                                                                         this->totalSumVector_, tmpVector_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::calculateScoresUncoveredAccumulated`
                     */
                    const IScoreVector& calculateScoresUncoveredAccumulated() override {
                        tmpVector_.difference(subsetSumVector_->cbegin(), subsetSumVector_->cend(), this->labelIndices_,
                                              accumulatedSumVectorPtr_->cbegin(), accumulatedSumVectorPtr_->cend());
                        return this->ruleEvaluationPtr_->calculateScores(this->majorityLabelVector_,
                                                                         this->totalSumVector_, tmpVector_);
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
             *                              predictions of rules, as well as their overall quality
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
                initializeLabelWiseStatisticVector(weights, labelMatrix, majorityLabelVector, coverageMatrix,
                                                   totalSumVector_);
                initializeLabelWiseStatisticVector(weights, labelMatrix, majorityLabelVector, coverageMatrix,
                                                   subsetSumVector_);
            }

            /**
             * @param statistics A reference to an object of type `LabelWiseWeightedStatistics` to be copied
             */
            LabelWiseWeightedStatistics(const LabelWiseWeightedStatistics& statistics)
                : weights_(statistics.weights_), ruleEvaluationFactory_(statistics.ruleEvaluationFactory_),
                  labelMatrix_(statistics.labelMatrix_), majorityLabelVector_(statistics.majorityLabelVector_),
                  totalSumVector_(ConfusionMatrixVector(statistics.totalSumVector_)),
                  subsetSumVector_(ConfusionMatrixVector(statistics.subsetSumVector_)),
                  coverageMatrix_(statistics.coverageMatrix_) {}

            /**
             * @see `IImmutableWeightedStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override {
                return labelMatrix_.getNumRows();
            }

            /**
             * @see `IImmutableWeightedStatistics::getNumLabels`
             */
            uint32 getNumLabels() const override {
                return labelMatrix_.getNumCols();
            }

            /**
             * @see `IWeightedStatistics::copy`
             */
            std::unique_ptr<IWeightedStatistics> copy() const override {
                return std::make_unique<LabelWiseWeightedStatistics<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                                    RuleEvaluationFactory, WeightVector>>(*this);
            }

            /**
             * @see `IWeightedStatistics::resetCoveredStatistics`
             */
            void resetCoveredStatistics() override {
                subsetSumVector_.clear();
            }

            /**
             * @see `IWeightedStatistics::addCoveredStatistic`
             */
            void addCoveredStatistic(uint32 statisticIndex) override {
                addLabelWiseStatistic(weights_, labelMatrix_, majorityLabelVector_, coverageMatrix_, subsetSumVector_,
                                      statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::removeCoveredStatistic`
             */
            void removeCoveredStatistic(uint32 statisticIndex) override {
                removeLabelWiseStatistic(weights_, labelMatrix_, majorityLabelVector_, coverageMatrix_,
                                         subsetSumVector_, statisticIndex);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
              const CompleteIndexVector& labelIndices) const override {
                return std::make_unique<WeightedStatisticsSubset<CompleteIndexVector>>(*this, labelIndices);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
              const PartialIndexVector& labelIndices) const override {
                return std::make_unique<WeightedStatisticsSubset<PartialIndexVector>>(*this, labelIndices);
            }

            /**
             * @see `IWeightedStatistics::createHistogram`
             */
            std::unique_ptr<IHistogram> createHistogram(const DenseBinIndexVector& binIndexVector,
                                                        uint32 numBins) const override {
                // TODO Support creation of histograms
                return nullptr;
            }

            /**
             * @see `IWeightedStatistics::createHistogram`
             */
            std::unique_ptr<IHistogram> createHistogram(const DokBinIndexVector& binIndexVector,
                                                        uint32 numBins) const override {
                // TODO Support creation of histograms
                return nullptr;
            }
    };

    template<typename Prediction, typename CoverageMatrix>
    static inline void applyLabelWisePredictionInternally(uint32 statisticIndex, const Prediction& prediction,
                                                          CoverageMatrix& coverageMatrix,
                                                          const VectorConstView<uint32>& majorityLabelIndices) {
        coverageMatrix.increaseCoverage(statisticIndex, majorityLabelIndices, prediction.scores_cbegin(),
                                        prediction.scores_cend(), prediction.indices_cbegin(),
                                        prediction.indices_cend());
    }

    template<typename Prediction, typename CoverageMatrix>
    static inline void revertLabelWisePredictionInternally(uint32 statisticIndex, const Prediction& prediction,
                                                           CoverageMatrix& coverageMatrix,
                                                           const VectorConstView<uint32>& majorityLabelIndices) {
        coverageMatrix.decreaseCoverage(statisticIndex, majorityLabelIndices, prediction.scores_cbegin(),
                                        prediction.scores_cend(), prediction.indices_cbegin(),
                                        prediction.indices_cend());
    }

    template<typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector,
             typename RuleEvaluationFactory, typename WeightVector, typename IndexVector>
    static inline std::unique_ptr<IStatisticsSubset> createStatisticsSubsetInternally(
      const LabelMatrix& labelMatrix, const CoverageMatrix& coverageMatrix,
      const BinarySparseArrayVector& majorityLabelVector, const RuleEvaluationFactory& ruleEvaluationFactory,
      const WeightVector& weights, const IndexVector& labelIndices) {
        std::unique_ptr<ConfusionMatrixVector> totalSumVectorPtr =
          std::make_unique<ConfusionMatrixVector>(labelMatrix.getNumRows(), true);
        return std::make_unique<LabelWiseStatisticsSubset<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                          RuleEvaluationFactory, WeightVector, IndexVector>>(
          std::move(totalSumVectorPtr), labelMatrix, coverageMatrix, majorityLabelVector, ruleEvaluationFactory,
          weights, labelIndices);
    }

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

            const std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr_;

            const std::unique_ptr<CoverageMatrix> coverageMatrixPtr_;

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
                  coverageMatrixPtr_(std::move(coverageMatrixPtr)) {}

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
                applyLabelWisePredictionInternally<CompletePrediction, CoverageMatrix>(
                  statisticIndex, prediction, *coverageMatrixPtr_, *majorityLabelVectorPtr_);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                applyLabelWisePredictionInternally<PartialPrediction, CoverageMatrix>(
                  statisticIndex, prediction, *coverageMatrixPtr_, *majorityLabelVectorPtr_);
            }

            /**
             * @see `IStatistics::revertPrediction`
             */
            void revertPrediction(uint32 statisticIndex, const CompletePrediction& prediction) override final {
                revertLabelWisePredictionInternally<CompletePrediction, CoverageMatrix>(
                  statisticIndex, prediction, *coverageMatrixPtr_, *majorityLabelVectorPtr_);
            }

            /**
             * @see `IStatistics::revertPrediction`
             */
            void revertPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                revertLabelWisePredictionInternally<PartialPrediction, CoverageMatrix>(
                  statisticIndex, prediction, *coverageMatrixPtr_, *majorityLabelVectorPtr_);
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                // TODO Support evaluation of predictions
                return 0;
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& labelIndices,
                                                            const EqualWeightVector& weights) const override final {
                return createStatisticsSubsetInternally<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                        RuleEvaluationFactory, EqualWeightVector, CompleteIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices,
                                                            const EqualWeightVector& weights) const override final {
                return createStatisticsSubsetInternally<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                        RuleEvaluationFactory, EqualWeightVector, PartialIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& labelIndices,
                                                            const BitWeightVector& weights) const override final {
                return createStatisticsSubsetInternally<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                        RuleEvaluationFactory, BitWeightVector, CompleteIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices,
                                                            const BitWeightVector& weights) const override final {
                return createStatisticsSubsetInternally<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                        RuleEvaluationFactory, BitWeightVector, PartialIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& labelIndices, const DenseWeightVector<uint32>& weights) const override final {
                return createStatisticsSubsetInternally<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                        RuleEvaluationFactory, DenseWeightVector<uint32>,
                                                        CompleteIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& labelIndices, const DenseWeightVector<uint32>& weights) const override final {
                return createStatisticsSubsetInternally<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                        RuleEvaluationFactory, DenseWeightVector<uint32>,
                                                        PartialIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& labelIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override final {
                return createStatisticsSubsetInternally<
                  LabelMatrix, CoverageMatrix, ConfusionMatrixVector, RuleEvaluationFactory,
                  OutOfSampleWeightVector<EqualWeightVector>, CompleteIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& labelIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override final {
                return createStatisticsSubsetInternally<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                        RuleEvaluationFactory,
                                                        OutOfSampleWeightVector<EqualWeightVector>, PartialIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& labelIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override final {
                return createStatisticsSubsetInternally<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                        RuleEvaluationFactory, OutOfSampleWeightVector<BitWeightVector>,
                                                        CompleteIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& labelIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override final {
                return createStatisticsSubsetInternally<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                        RuleEvaluationFactory, OutOfSampleWeightVector<BitWeightVector>,
                                                        PartialIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& labelIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override final {
                return createStatisticsSubsetInternally<
                  LabelMatrix, CoverageMatrix, ConfusionMatrixVector, RuleEvaluationFactory,
                  OutOfSampleWeightVector<DenseWeightVector<uint32>>, CompleteIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& labelIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override final {
                return createStatisticsSubsetInternally<
                  LabelMatrix, CoverageMatrix, ConfusionMatrixVector, RuleEvaluationFactory,
                  OutOfSampleWeightVector<DenseWeightVector<uint32>>, PartialIndexVector>(
                  labelMatrix_, *coverageMatrixPtr_, *majorityLabelVectorPtr_, *ruleEvaluationFactory_, weights,
                  labelIndices);
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
