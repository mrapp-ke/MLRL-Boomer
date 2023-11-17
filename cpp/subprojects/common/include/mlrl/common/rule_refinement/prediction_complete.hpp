/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/array.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/rule_refinement/prediction_evaluated.hpp"

/**
 * Stores the scores that are predicted by a rule that predicts for all available labels.
 */
class CompletePrediction final : public VectorDecorator<AllocatedVector<float64>>,
                                 public IEvaluatedPrediction {
    private:

        const CompleteIndexVector indexVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        CompletePrediction(uint32 numElements);

        /**
         * An iterator that provides access to the predicted scores and allows to modify them.
         */
        typedef View<float64>::iterator value_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores.
         */
        typedef View<float64>::const_iterator value_const_iterator;

        /**
         * An iterator that provides read-only access to the indices of the labels for which the rule predicts.
         */
        typedef CompleteIndexVector::const_iterator index_const_iterator;

        /**
         * Returns a `value_iterator` to the beginning of the predicted scores.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin();

        /**
         * Returns a `value_iterator` to the end of the predicted scores.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end();

        /**
         * Returns a `value_const_iterator` to the beginning of the predicted scores.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the predicted scores.
         *
         * @return A `const_iterator` to the end
         */
        value_const_iterator values_cend() const;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices of the labels for which the rule predicts.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices of the labels for which the rule predicts.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        uint32 getNumElements() const override;

        void sort() override;

        void postProcess(const IPostProcessor& postProcessor) override;

        void set(View<float64>::const_iterator begin, View<float64>::const_iterator end) override final;

        void set(DenseBinnedVector<float64>::const_iterator begin,
                 DenseBinnedVector<float64>::const_iterator end) override final;

        bool isPartial() const override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(const IStatistics& statistics,
                                                                  const EqualWeightVector& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(const IStatistics& statistics,
                                                                  const BitWeightVector& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const DenseWeightVector<uint32>& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<EqualWeightVector>& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<BitWeightVector>& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics,
          const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override;

        std::unique_ptr<IRuleRefinement> createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                              uint32 featureIndex) const override;

        void apply(IStatistics& statistics, uint32 statisticIndex) const override;

        void revert(IStatistics& statistics, uint32 statisticIndex) const override;

        std::unique_ptr<IHead> createHead() const override;
};
