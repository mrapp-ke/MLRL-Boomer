/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_state.hpp"
#include "mlrl/common/statistics/statistics_update_candidate.hpp"
#include "mlrl/common/util/concepts.hpp"

/**
 * A base class for all classes that store scores that have been calculated based on statistics and allow to update
 * these statistics accordingly.
 */
class AbstractStatisticsUpdateCandidate : public IStatisticsUpdateCandidate {
    private:

        const IScoreVector& scoreVector_;

    protected:

        /**
         * Allows updating the statistics.
         *
         * @tparam State        The type of the state of the statistics
         * @tparam IndexVector  The type of the vector that provides access to the indices of the outputs for which
         *                      confusion matrices should be updated
         */
        template<util::derived_from_template_class<IStatisticsState> State, std::derived_from<IIndexVector> IndexVector>
        class StatisticsUpdate final : public IStatisticsUpdate {
            private:

                State& state_;

                typename IndexVector::const_iterator indicesBegin_;

                typename IndexVector::const_iterator indicesEnd_;

                typename View<typename State::score_type>::const_iterator scoresBegin_;

                typename View<typename State::score_type>::const_iterator scoresEnd_;

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
                                 typename View<typename State::score_type>::const_iterator scoresBegin,
                                 typename View<typename State::score_type>::const_iterator scoresEnd)
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
        template<util::derived_from_template_class<IStatisticsState> State>
        class StatisticsUpdateFactory final : public IStatisticsUpdateFactory<typename State::score_type> {
            private:

                State& state_;

            public:

                /**
                 * @param state A reference to an object of template type `State` that should be updated
                 */
                StatisticsUpdateFactory(State& state) : state_(state) {}

                std::unique_ptr<IStatisticsUpdate> create(
                  CompleteIndexVector::const_iterator indicesBegin, CompleteIndexVector::const_iterator indicesEnd,
                  typename View<typename State::score_type>::const_iterator scoresBegin,
                  typename View<typename State::score_type>::const_iterator scoresEnd) override {
                    return std::make_unique<StatisticsUpdate<State, CompleteIndexVector>>(
                      state_, indicesBegin, indicesEnd, scoresBegin, scoresEnd);
                }

                std::unique_ptr<IStatisticsUpdate> create(
                  PartialIndexVector::const_iterator indicesBegin, PartialIndexVector::const_iterator indicesEnd,
                  typename View<typename State::score_type>::const_iterator scoresBegin,
                  typename View<typename State::score_type>::const_iterator scoresEnd) override {
                    return std::make_unique<StatisticsUpdate<State, PartialIndexVector>>(
                      state_, indicesBegin, indicesEnd, scoresBegin, scoresEnd);
                }
        };

        /**
         * May be overridden by subclasses in order to invoke a given `DenseVisitor` for handling objects of type
         * `DenseScoreVector<float32, CompleteIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseScoreVector<float32, CompleteIndexVector>` to be
         *                      handled by the visitor
         */
        virtual void invokeVisitor(DenseVisitor<float32, CompleteIndexVector> visitor,
                                   const DenseScoreVector<float32, CompleteIndexVector>& scoreVector) const {
            throw std::runtime_error("not implemented");
        }

        /**
         * May be overridden by subclasses in order to invoke a given `DenseVisitor` for handling objects of type
         * `DenseScoreVector<float32, PartialIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseScoreVector<float32, PartialIndexVector>` to be
         *                      handled by the visitor
         */
        virtual void invokeVisitor(DenseVisitor<float32, PartialIndexVector> visitor,
                                   const DenseScoreVector<float32, PartialIndexVector>& scoreVector) const {
            throw std::runtime_error("not implemented");
        }

        /**
         * May be overridden by subclasses in order to invoke a given `DenseVisitor` for handling objects of type
         * `DenseScoreVector<float64, CompleteIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseScoreVector<float64, CompleteIndexVector>` to be
         *                      handled by the visitor
         */
        virtual void invokeVisitor(DenseVisitor<float64, CompleteIndexVector> visitor,
                                   const DenseScoreVector<float64, CompleteIndexVector>& scoreVector) const {
            throw std::runtime_error("not implemented");
        }

        /**
         * May be overridden by subclasses in order to invoke a given `DenseVisitor` for handling objects of type
         * `DenseScoreVector<float64, PartialIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseScoreVector<float64, PartialIndexVector>` to be
         *                      handled by the visitor
         */
        virtual void invokeVisitor(DenseVisitor<float64, PartialIndexVector> visitor,
                                   const DenseScoreVector<float64, PartialIndexVector>& scoreVector) const {
            throw std::runtime_error("not implemented");
        }

        /**
         * May be overridden by subclasses in order to invoke a given `DenseBinnedVisitor` for handling objects of type
         * `DenseBinnedScoreVector<float32, CompleteIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseBinnedScoreVector<float32, CompleteIndexVector>`
         *                      to be handled by the visitor
         */
        virtual void invokeVisitor(DenseBinnedVisitor<float32, CompleteIndexVector> visitor,
                                   const DenseBinnedScoreVector<float32, CompleteIndexVector>& scoreVector) const {
            throw std::runtime_error("not implemented");
        }

        /**
         * May be overridden by subclasses in order to invoke a given `DenseBinnedVisitor` for handling objects of type
         * `DenseBinnedScoreVector<float32, PartialIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseBinnedScoreVector<float32, PartialIndexVector>`
         *                      to be handled by the visitor
         */
        virtual void invokeVisitor(DenseBinnedVisitor<float32, PartialIndexVector> visitor,
                                   const DenseBinnedScoreVector<float32, PartialIndexVector>& scoreVector) const {
            throw std::runtime_error("not implemented");
        }

        /**
         * May be overridden by subclasses in order to invoke a given `DenseBinnedVisitor` for handling objects of type
         * `DenseBinnedScoreVector<float64, CompleteIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseBinnedScoreVector<float64, CompleteIndexVector>`
         *                      to be handled by the visitor
         */
        virtual void invokeVisitor(DenseBinnedVisitor<float64, CompleteIndexVector> visitor,
                                   const DenseBinnedScoreVector<float64, CompleteIndexVector>& scoreVector) const {
            throw std::runtime_error("not implemented");
        }

        /**
         * May be overridden by subclasses in order to invoke a given `DenseBinnedVisitor` for handling objects of type
         * `DenseBinnedScoreVector<float64, PartialIndexVector>`.
         *
         * @param visitor       The visitor to be invoked
         * @param scoreVector   A reference to an object of type `DenseBinnedScoreVector<float64, PartialIndexVector>`
         *                      to be handled by the visitor
         */
        virtual void invokeVisitor(DenseBinnedVisitor<float64, PartialIndexVector> visitor,
                                   const DenseBinnedScoreVector<float64, PartialIndexVector>& scoreVector) const {
            throw std::runtime_error("not implemented");
        }

    public:

        /**
         * @param scoreVector A reference to an object of type `IScoreVector` that stores the calculated scores
         */
        explicit AbstractStatisticsUpdateCandidate(const IScoreVector& scoreVector) : scoreVector_(scoreVector) {
            this->quality = scoreVector.quality;
        }

        virtual ~AbstractStatisticsUpdateCandidate() override {}

        void visit(
          DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
          DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
          DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
          DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
          DenseBinnedVisitor<float32, CompleteIndexVector> completeDenseBinned32BitVisitor,
          DenseBinnedVisitor<float32, PartialIndexVector> partialDenseBinned32BitVisitor,
          DenseBinnedVisitor<float64, CompleteIndexVector> completeDenseBinned64BitVisitor,
          DenseBinnedVisitor<float64, PartialIndexVector> partialDenseBinned64BitVisitor) const override final {
            auto tmpCompleteDense32BitVisitor =
              [this, completeDense32BitVisitor](const DenseScoreVector<float32, CompleteIndexVector>& scoreVector) {
                invokeVisitor(completeDense32BitVisitor, scoreVector);
            };
            auto tmpPartialDense32BitVisitor =
              [this, partialDense32BitVisitor](const DenseScoreVector<float32, PartialIndexVector>& scoreVector) {
                invokeVisitor(partialDense32BitVisitor, scoreVector);
            };
            auto tmpCompleteDense64BitVisitor =
              [this, completeDense64BitVisitor](const DenseScoreVector<float64, CompleteIndexVector>& scoreVector) {
                invokeVisitor(completeDense64BitVisitor, scoreVector);
            };
            auto tmpPartialDense64BitVisitor =
              [this, partialDense64BitVisitor](const DenseScoreVector<float64, PartialIndexVector>& scoreVector) {
                invokeVisitor(partialDense64BitVisitor, scoreVector);
            };
            auto tmpCompleteDenseBinned32BitVisitor =
              [this, completeDenseBinned32BitVisitor](
                const DenseBinnedScoreVector<float32, CompleteIndexVector>& scoreVector) {
                invokeVisitor(completeDenseBinned32BitVisitor, scoreVector);
            };
            auto tmpPartialDenseBinned32BitVisitor =
              [this,
               partialDenseBinned32BitVisitor](const DenseBinnedScoreVector<float32, PartialIndexVector>& scoreVector) {
                invokeVisitor(partialDenseBinned32BitVisitor, scoreVector);
            };
            auto tmpCompleteDenseBinned64BitVisitor =
              [this, completeDenseBinned64BitVisitor](
                const DenseBinnedScoreVector<float64, CompleteIndexVector>& scoreVector) {
                invokeVisitor(completeDenseBinned64BitVisitor, scoreVector);
            };
            auto tmpPartialDenseBinned64BitVisitor =
              [this,
               partialDenseBinned64BitVisitor](const DenseBinnedScoreVector<float64, PartialIndexVector>& scoreVector) {
                invokeVisitor(partialDenseBinned64BitVisitor, scoreVector);
            };
            scoreVector_.visit(tmpCompleteDense32BitVisitor, tmpPartialDense32BitVisitor, tmpCompleteDense64BitVisitor,
                               tmpPartialDense64BitVisitor, tmpCompleteDenseBinned32BitVisitor,
                               tmpPartialDenseBinned32BitVisitor, tmpCompleteDenseBinned64BitVisitor,
                               tmpPartialDenseBinned64BitVisitor);
        }
};
