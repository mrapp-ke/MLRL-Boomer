/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_update_candidate_common.hpp"

namespace boosting {

    /**
     * Stores scores that have been calculated based on gradients and Hessians, represented by 32-bit floating point
     * values, and allow to update these gradients and Hessians accordingly.
     *
     * @tparam State The type of the state of the boosting process
     */
    template<util::derived_from_template_class<IStatisticsState> State>
    class Boosting32BitStatisticsUpdateCandidate final : public AbstractStatisticsUpdateCandidate {
        private:

            State& state_;

        protected:

            void invokeVisitor(DenseVisitor<float32, CompleteIndexVector> visitor,
                               const DenseScoreVector<float32, CompleteIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

            void invokeVisitor(DenseVisitor<float32, PartialIndexVector> visitor,
                               const DenseScoreVector<float32, PartialIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

            void invokeVisitor(DenseBinnedVisitor<float32, CompleteIndexVector> visitor,
                               const DenseBinnedScoreVector<float32, CompleteIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

            void invokeVisitor(DenseBinnedVisitor<float32, PartialIndexVector> visitor,
                               const DenseBinnedScoreVector<float32, PartialIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

        public:

            /**
             * @param state         A reference to an object of template type `State` that represents the state of the
             *                      boosting process
             * @param scoreVector   A reference to an object of type `IScoreVector` that stores the calculated scores
             */
            Boosting32BitStatisticsUpdateCandidate(State& state, const IScoreVector& scoreVector)
                : AbstractStatisticsUpdateCandidate(scoreVector), state_(state) {}
    };

    /**
     * Stores scores that have been calculated based on gradients and Hessians, represented by 64-bit floating point
     * values,  and allow to update these gradients and Hessians accordingly.
     *
     * @tparam State The type of the state of the boosting process
     */
    template<util::derived_from_template_class<IStatisticsState> State>
    class Boosting64BitStatisticsUpdateCandidate final : public AbstractStatisticsUpdateCandidate {
        private:

            State& state_;

        protected:

            void invokeVisitor(DenseVisitor<float64, CompleteIndexVector> visitor,
                               const DenseScoreVector<float64, CompleteIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

            void invokeVisitor(DenseVisitor<float64, PartialIndexVector> visitor,
                               const DenseScoreVector<float64, PartialIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

            void invokeVisitor(DenseBinnedVisitor<float64, CompleteIndexVector> visitor,
                               const DenseBinnedScoreVector<float64, CompleteIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

            void invokeVisitor(DenseBinnedVisitor<float64, PartialIndexVector> visitor,
                               const DenseBinnedScoreVector<float64, PartialIndexVector>& scoreVector) const override {
                StatisticsUpdateFactory<State> statisticsUpdateFactory(state_);
                visitor(scoreVector, statisticsUpdateFactory);
            }

        public:

            /**
             * @param state         A reference to an object of template type `State` that represents the state of the
             *                      boosting process
             * @param scoreVector   A reference to an object of type `IScoreVector` that stores the calculated scores
             */
            Boosting64BitStatisticsUpdateCandidate(State& state, const IScoreVector& scoreVector)
                : AbstractStatisticsUpdateCandidate(scoreVector), state_(state) {}
    };

}
