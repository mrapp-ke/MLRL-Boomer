/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_update_candidate.hpp"

namespace boosting {

    /**
     * Stores scores that have been calculated based on gradients and Hessians and allow to update them accordingly.
     *
     * @tparam State The type of the state of the boosting process
     */
    template<std::derived_from<IStatisticsState> State>
    class BoostingStatisticsUpdateCandidate final : public StatisticsUpdateCandidate {
        private:

            State& state_;

        protected:

            void invokeVisitor(DenseVisitor<CompleteIndexVector> visitor,
                               const DenseScoreVector<CompleteIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

            void invokeVisitor(DenseVisitor<PartialIndexVector> visitor,
                               const DenseScoreVector<PartialIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

            void invokeVisitor(DenseBinnedVisitor<CompleteIndexVector> visitor,
                               const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

            void invokeVisitor(DenseBinnedVisitor<PartialIndexVector> visitor,
                               const DenseBinnedScoreVector<PartialIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

        public:

            /**
             * @param state         A reference to an object of template type `State` that represents the state of the
             *                      boosting process
             * @param scoreVector   A reference to an object of type `IScoreVector` that stores the calculated scores
             */
            BoostingStatisticsUpdateCandidate(State& state, const IScoreVector& scoreVector)
                : StatisticsUpdateCandidate(scoreVector), state_(state) {}
    };

}
