/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_refinement/prediction_evaluated.hpp"

#include <memory>

/**
 * Stores the scores that are predicted by a rule that predicts for a subset of the available outputs.
 */
class PartialPrediction final : public VectorDecorator<ResizableVector<float64>>,
                                public IEvaluatedPrediction {
    private:

        PartialIndexVector indexVector_;

        bool sorted_;

        std::unique_ptr<IStatisticsUpdate> statisticsUpdatePtr_;

    public:

        /**
         * @param numElements             The number of outputs for which the rule predicts
         * @param sorted                  True, if the scores that are stored by this prediction are sorted in
         *                                increasing order by the corresponding output indices, false otherwise
         * @param statisticsUpdateFactory A reference to an object of type `IStatisticsUpdateFactory`
         */
        PartialPrediction(uint32 numElements, bool sorted, IStatisticsUpdateFactory<float64>& statisticsUpdateFactory);

        /**
         * An iterator that provides access to the predicted scores and allows to modify them.
         */
        typedef View<float64>::iterator value_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores.
         */
        typedef View<float64>::const_iterator value_const_iterator;

        /**
         * An iterator that provides access to the indices for which the rule predicts and allows to modify them.
         */
        typedef PartialIndexVector::iterator index_iterator;

        /**
         * An iterator that provides read-only access to the indices for which the rule predicts.
         */
        typedef PartialIndexVector::const_iterator index_const_iterator;

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
         * Returns an `index_iterator` to the beginning of the indices for which the rule predicts.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin();

        /**
         * Returns an `index_iterator` to the end of the indices for which the rule predicts.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the indices for which the rule predicts.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices for which the rule predicts.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Sets whether the scores that are stored by this prediction are sorted in increasing order by the
         * corresponding output indices, or not.
         *
         * @param sorted True, if the scores that are stored by this prediction are sorted in increasing order by the
         *               corresponding output indices, false otherwise
         */
        void setSorted(bool sorted);

        /**
         * Sets the number of elements in the vector.
         *
         * @param statisticsUpdateFactory A reference to an object of type `IStatisticsUpdateFactory`
         * @param numElements             The number of elements to be set
         * @param freeMemory              True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumElements(IStatisticsUpdateFactory<float64>& statisticsUpdateFactory, uint32 numElements,
                            bool freeMemory);

        uint32 getNumElements() const override;

        void sort() override;

        void postProcess(const IPostProcessor& postProcessor) override;

        bool isPartial() const override;

        uint32 getIndex(uint32 pos) const override;

        void visit(PartialIndexVectorVisitor partialIndexVectorVisitor,
                   CompleteIndexVectorVisitor completeIndexVectorVisitor) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(const IStatistics& statistics,
                                                                  const EqualWeightVector& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(const IStatistics& statistics,
                                                                  const BitWeightVector& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const DenseWeightVector<uint32>& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const DenseWeightVector<float32>& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<EqualWeightVector>& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<BitWeightVector>& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics,
          const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics,
          const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override;

        void applyPrediction(uint32 statisticIndex) override;

        void revertPrediction(uint32 statisticIndex) override;

        std::unique_ptr<IHead> createHead() const override;
};
