/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"
#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "mlrl/common/statistics/statistics_state.hpp"
#include "mlrl/common/statistics/statistics_update.hpp"

#include <concepts>

/**
 * A base class for all classes that store scores that have been calculated based on statistics and allow to update
 * them accordingly.
 */
class StatisticsUpdateCandidate : public Quality {
    protected:

        /**
         * Allows updating the statistics.
         *
         * @tparam State        The type of the state of the statistics
         * @tparam IndexVector  The type of the vector that provides access to the indices of the outputs for which
         *                      confusion matrices should be updated
         */
        template<std::derived_from<IStatisticsState> State, std::derived_from<IIndexVector> IndexVector>
        class StatisticsUpdate final : public IStatisticsUpdate {
            private:

                State& state_;

                typename IndexVector::const_iterator indicesBegin_;

                typename IndexVector::const_iterator indicesEnd_;

                View<float64>::const_iterator scoresBegin_;

                View<float64>::const_iterator scoresEnd_;

            public:

                /**
                 * @param state         A reference to an object of template type `State` that should be updated
                 * @param indicesBegin  An iterator to the beginning of the output indices for which statistics
                 *                      should be updated
                 * @param indicesEnd    An iterator to the end of the output indices for which statistics should be
                 *                      updated
                 * @param scoresBegin   An iterator to the beginning of the predicted scores, corresponding to the
                 *                      given output indices, that should be used for updating the statistics
                 * @param scoresEnd     An iterator to the end of the predicted scores, corresponding to the given
                 *                      output indices, that should be used for updating the statistics
                 */
                StatisticsUpdate(State& state, typename IndexVector::const_iterator indicesBegin,
                                 typename IndexVector::const_iterator indicesEnd,
                                 View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd)
                    : state_(state), indicesBegin_(indicesBegin), indicesEnd_(indicesEnd), scoresBegin_(scoresBegin),
                      scoresEnd_(scoresEnd) {}

                void applyPrediction(uint32 statisticIndex) override {
                    state_.update(statisticIndex, scoresBegin_, scoresEnd_, indicesBegin_, indicesEnd_);
                }

                void revertPrediction(uint32 statisticIndex) override {
                    state_.revert(statisticIndex, scoresBegin_, scoresEnd_, indicesBegin_, indicesEnd_);
                }
        };

        /**
         * Allows to create instances of the type `IStatisticsUpdate` that allow updating the statistics.

         * @tparam State The type of the state of the statistics
         */
        template<std::derived_from<IStatisticsState> State>
        class StatisticsUpdateFactory final : public IStatisticsUpdateFactory {
            private:

                State& state_;

            public:

                /**
                 * @param state A reference to an object of template type `State` that should be updated
                 */
                StatisticsUpdateFactory(State& state) : state_(state) {}

                std::unique_ptr<IStatisticsUpdate> create(CompleteIndexVector::const_iterator indicesBegin,
                                                          CompleteIndexVector::const_iterator indicesEnd,
                                                          View<float64>::const_iterator scoresBegin,
                                                          View<float64>::const_iterator scoresEnd) override {
                    return std::make_unique<StatisticsUpdate<State, CompleteIndexVector>>(
                      state_, indicesBegin, indicesEnd, scoresBegin, scoresEnd);
                }

                std::unique_ptr<IStatisticsUpdate> create(PartialIndexVector::const_iterator indicesBegin,
                                                          PartialIndexVector::const_iterator indicesEnd,
                                                          View<float64>::const_iterator scoresBegin,
                                                          View<float64>::const_iterator scoresEnd) override {
                    return std::make_unique<StatisticsUpdate<State, PartialIndexVector>>(
                      state_, indicesBegin, indicesEnd, scoresBegin, scoresEnd);
                }
        };

    private:

        const IScoreVector& scoreVector_;

    public:

        /**
         * A visitor function for handling objects of type `DenseScoreVector`.
         *
         * @tparam IndexVector The type of the vector that provides access to the indices of the outputs, the predicted
         *                     scores correspond to
         */
        template<typename IndexVector>
        using DenseVisitor = std::function<void(const DenseScoreVector<IndexVector>&, IStatisticsUpdateFactory&)>;

        /**
         * A visitor function for handling objects of type `DenseBinnedScoreVector`.
         *
         * @tparam IndexVector The type of the vector that provides access to the indices of the outputs, the predicted
         *                     scores correspond to
         */
        template<typename IndexVector>
        using DenseBinnedVisitor =
          std::function<void(const DenseBinnedScoreVector<IndexVector>&, IStatisticsUpdateFactory&)>;

    protected:

        /**
         * May be overridden by subclasses in order to invoke a given `DenseVisitor` for handling objects of type
         * `DenseScoreVector<CompleteIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseScoreVector<CompleteIndexVector>` to be handled
         *                      by the visitor
         */
        virtual void invokeVisitor(DenseVisitor<CompleteIndexVector> visitor,
                                   const DenseScoreVector<CompleteIndexVector>& scoreVector) const;

        /**
         * May be overridden by subclasses in order to invoke a given `DenseVisitor` for handling objects of type
         * `DenseScoreVector<PartialIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseScoreVector<PartialIndexVector>` to be handled by
         *                      the visitor
         */
        virtual void invokeVisitor(DenseVisitor<PartialIndexVector> visitor,
                                   const DenseScoreVector<PartialIndexVector>& scoreVector) const;

        /**
         * May be overridden by subclasses in order to invoke a given `DenseBinnedVisitor` for handling objects of type
         * `DenseBinnedScoreVector<CompleteIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseBinnedScoreVector<CompleteIndexVector>` to be
         *                      handled by the visitor
         */
        virtual void invokeVisitor(DenseBinnedVisitor<CompleteIndexVector> visitor,
                                   const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector) const;

        /**
         * May be overridden by subclasses in order to invoke a given `DenseBinnedVisitor` for handling objects of type
         * `DenseBinnedScoreVector<PartialIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseBinnedScoreVector<PartialIndexVector>` to be
         *                      handled by the visitor
         */
        virtual void invokeVisitor(DenseBinnedVisitor<PartialIndexVector> visitor,
                                   const DenseBinnedScoreVector<PartialIndexVector>& scoreVector) const;

    public:

        /**
         * @param scoreVector A reference to an object of type `IScoreVector` that stores the calculated scores
         */
        explicit StatisticsUpdateCandidate(const IScoreVector& scoreVector);

        virtual ~StatisticsUpdateCandidate() {}

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle the particular type of
         * vector that stores the calculated scores.
         *
         * @param completeDenseVisitor          The visitor function for handling objects of type
         *                                      `DenseScoreVector<CompleteIndexVector>`
         * @param partialDenseVisitor           The visitor function for handling objects of type
         *                                      `DenseScoreVector<PartialIndexVector>`
         * @param completeDenseBinnedVisitor    The visitor function for handling objects of type
         *                                      `DenseBinnedScoreVector<CompleteIndexVector>`
         * @param partialDenseBinnedVisitor     The visitor function for handling objects of type
         *                                      `DenseBinnedScoreVector<PartialIndexVector>`
         */
        void visit(DenseVisitor<CompleteIndexVector> completeDenseVisitor,
                   DenseVisitor<PartialIndexVector> partialDenseVisitor,
                   DenseBinnedVisitor<CompleteIndexVector> completeDenseBinnedVisitor,
                   DenseBinnedVisitor<PartialIndexVector> partialDenseBinnedVisitor) const;
};
