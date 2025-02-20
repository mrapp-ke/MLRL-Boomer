/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_update_candidate_common.hpp"

#include <memory>

namespace boosting {

    /**
     * Represents the current state of a sequential boosting process and allows to update it.
     *
     * The state consists of gradients and Hessians that correspond to the quality of a model's predictions for the
     * training examples. When the model has been modified, this state can be updated accordingly by recalculating
     * affected gradients and Hessians via a loss function that operates on the scores that are predicted by the updated
     * model and the corresponding ground truth of the training examples.
     *
     * @tparam OutputMatrix     The type of the matrix that provides access to the ground truth of the training examples
     * @tparam StatisticMatrix  The type of the matrix that provides access to the gradients and Hessians
     * @tparam ScoreMatrix      The type of the matrix that is used to store predicted scores
     * @tparam Loss             The type of the loss function that is used to calculate gradients and Hessians
     */
    template<typename OutputMatrix, typename StatisticMatrix, typename ScoreMatrix, typename Loss>
    class AbstractBoostingStatisticsState : public IStatisticsState<typename ScoreMatrix::value_type> {
        private:

            /**
             * Stores scores that have been calculated based on gradients and Hessians, represented by 32-bit floating
             * point values, and allows to update these gradients and Hessians accordingly.
             */
            class Float32UpdateCandidate final : public AbstractStatisticsUpdateCandidate {
                private:

                    IStatisticsState<float32>& state_;

                protected:

                    void invokeVisitor(
                      DenseVisitor<float32, CompleteIndexVector> visitor,
                      const DenseScoreVector<float32, CompleteIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<IStatisticsState<float32>> statisticsUpdateFactory(state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                    void invokeVisitor(
                      DenseVisitor<float32, PartialIndexVector> visitor,
                      const DenseScoreVector<float32, PartialIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<IStatisticsState<float32>> statisticsUpdateFactory(state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                    void invokeVisitor(
                      DenseBinnedVisitor<float32, CompleteIndexVector> visitor,
                      const DenseBinnedScoreVector<float32, CompleteIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<IStatisticsState<float32>> statisticsUpdateFactory(state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                    void invokeVisitor(
                      DenseBinnedVisitor<float32, PartialIndexVector> visitor,
                      const DenseBinnedScoreVector<float32, PartialIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<IStatisticsState<float32>> statisticsUpdateFactory(state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                public:

                    /**
                     * @param state         A reference to an object of template type `IStatisticsState<float32>` that
                     *                      represents the state of the boosting process
                     * @param scoreVector   A reference to an object of type `IScoreVector` that stores the calculated
                     *                      scores
                     */
                    Float32UpdateCandidate(IStatisticsState<float32>& state, const IScoreVector& scoreVector)
                        : AbstractStatisticsUpdateCandidate(scoreVector), state_(state) {}
            };

            /**
             * Stores scores that have been calculated based on gradients and Hessians, represented by 64-bit floating
             * point values, and allows to update these gradients and Hessians accordingly.
             */
            class Float64UpdateCandidate final : public AbstractStatisticsUpdateCandidate {
                private:

                    IStatisticsState<float64>& state_;

                protected:

                    void invokeVisitor(
                      DenseVisitor<float64, CompleteIndexVector> visitor,
                      const DenseScoreVector<float64, CompleteIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<IStatisticsState<float64>> statisticsUpdateFactory(state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                    void invokeVisitor(
                      DenseVisitor<float64, PartialIndexVector> visitor,
                      const DenseScoreVector<float64, PartialIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<IStatisticsState<float64>> statisticsUpdateFactory(state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                    void invokeVisitor(
                      DenseBinnedVisitor<float64, CompleteIndexVector> visitor,
                      const DenseBinnedScoreVector<float64, CompleteIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<IStatisticsState<float64>> statisticsUpdateFactory(state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                    void invokeVisitor(
                      DenseBinnedVisitor<float64, PartialIndexVector> visitor,
                      const DenseBinnedScoreVector<float64, PartialIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<IStatisticsState<float64>> statisticsUpdateFactory(state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                public:

                    /**
                     * @param state         A reference to an object of template type `IStatisticsState<float64>` that
                     *                      represents the state of the boosting process
                     * @param scoreVector   A reference to an object of type `IScoreVector` that stores the calculated
                     *                      scores
                     */
                    Float64UpdateCandidate(IStatisticsState<float64>& state, const IScoreVector& scoreVector)
                        : AbstractStatisticsUpdateCandidate(scoreVector), state_(state) {}
            };

            static inline std::unique_ptr<IStatisticsUpdateCandidate> createUpdateCandidateInternally(
              IStatisticsState<float32>& state, const IScoreVector& scoreVector) {
                return std::make_unique<Float32UpdateCandidate>(state, scoreVector);
            }

            static inline std::unique_ptr<IStatisticsUpdateCandidate> createUpdateCandidateInternally(
              IStatisticsState<float64>& state, const IScoreVector& scoreVector) {
                return std::make_unique<Float64UpdateCandidate>(state, scoreVector);
            }

        public:

            /**
             * The type of the scores that are used for updating the state.
             */
            typedef typename ScoreMatrix::value_type score_type;

            /**
             * A reference to an object of template type `OutputMatrix` that provides access to the ground truth of the
             * training examples.
             */
            const OutputMatrix& outputMatrix;

            /**
             * An unique pointer to an object of template type `StatisticMatrix` that stores the gradients and Hessians.
             */
            std::unique_ptr<StatisticMatrix> statisticMatrixPtr;

            /**
             * An unique pointer to an object of template type `ScoreMatrix` that stores the currently predicted scores.
             */
            std::unique_ptr<ScoreMatrix> scoreMatrixPtr;

            /**
             * An unique pointer to an object of template type `Loss` that is used for calculating gradients and
             * Hessians.
             */
            std::unique_ptr<Loss> lossFunctionPtr;

        protected:

            /**
             * Must be implemented by subclasses in order to update the statistics for all available outputs at a
             * specific index.
             *
             * @param statisticIndex    The index of the statistics to be updated
             * @param indicesBegin      An iterator to the beginning of the output indices
             * @param indicesEnd        An iterator to the end of the output indices
             */
            virtual void updateStatistics(uint32 statisticIndex, CompleteIndexVector::const_iterator indicesBegin,
                                          CompleteIndexVector::const_iterator indicesEnd) = 0;

            /**
             * Must be implemented by subclasses in order to update the statistics for a subset of the available outputs
             * at a specific index.
             *
             * @param statisticIndex    The index of the statistics to be updated
             * @param indicesBegin      An iterator to the beginning of the output indices
             * @param indicesEnd        An iterator to the end of the output indices
             */
            virtual void updateStatistics(uint32 statisticIndex, PartialIndexVector::const_iterator indicesBegin,
                                          PartialIndexVector::const_iterator indicesEnd) = 0;

        public:

            /**
             * @param outputMatrix          A reference to an object of template type `OutputMatrix` that provides
             *                              access to the ground truth of the training examples
             * @param statisticMatrixPtr    An unique pointer to an object of template type `StatisticMatrix` that
             *                              stores the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of template type `ScoreMatrix` that stores
             *                              the currently predicted scores
             * @param lossFunctionPtr       An unique pointer to the an object of template type `Loss` that should be
             *                              used for calculating gradients and Hessians
             */
            AbstractBoostingStatisticsState(const OutputMatrix& outputMatrix,
                                            std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                                            std::unique_ptr<ScoreMatrix> scoreMatrixPtr,
                                            std::unique_ptr<Loss> lossFunctionPtr)
                : outputMatrix(outputMatrix), statisticMatrixPtr(std::move(statisticMatrixPtr)),
                  scoreMatrixPtr(std::move(scoreMatrixPtr)), lossFunctionPtr(std::move(lossFunctionPtr)) {}

            virtual ~AbstractBoostingStatisticsState() {}

            void update(uint32 statisticIndex, typename View<score_type>::const_iterator scoresBegin,
                        typename View<score_type>::const_iterator scoresEnd,
                        CompleteIndexVector::const_iterator indicesBegin,
                        CompleteIndexVector::const_iterator indicesEnd) override final {
                scoreMatrixPtr->addToRowFromSubset(statisticIndex, scoresBegin, scoresEnd, indicesBegin, indicesEnd);
                updateStatistics(statisticIndex, indicesBegin, indicesEnd);
            }

            void update(uint32 statisticIndex, typename View<score_type>::const_iterator scoresBegin,
                        typename View<score_type>::const_iterator scoresEnd,
                        PartialIndexVector::const_iterator indicesBegin,
                        PartialIndexVector::const_iterator indicesEnd) override final {
                scoreMatrixPtr->addToRowFromSubset(statisticIndex, scoresBegin, scoresEnd, indicesBegin, indicesEnd);
                updateStatistics(statisticIndex, indicesBegin, indicesEnd);
            }

            void revert(uint32 statisticIndex, typename View<score_type>::const_iterator scoresBegin,
                        typename View<score_type>::const_iterator scoresEnd,
                        CompleteIndexVector::const_iterator indicesBegin,
                        CompleteIndexVector::const_iterator indicesEnd) override final {
                scoreMatrixPtr->removeFromRowFromSubset(statisticIndex, scoresBegin, scoresEnd, indicesBegin,
                                                        indicesEnd);
                updateStatistics(statisticIndex, indicesBegin, indicesEnd);
            }

            void revert(uint32 statisticIndex, typename View<score_type>::const_iterator scoresBegin,
                        typename View<score_type>::const_iterator scoresEnd,
                        PartialIndexVector::const_iterator indicesBegin,
                        PartialIndexVector::const_iterator indicesEnd) override final {
                scoreMatrixPtr->removeFromRowFromSubset(statisticIndex, scoresBegin, scoresEnd, indicesBegin,
                                                        indicesEnd);
                updateStatistics(statisticIndex, indicesBegin, indicesEnd);
            }

            std::unique_ptr<IStatisticsUpdateCandidate> createUpdateCandidate(
              const IScoreVector& scoreVector) override {
                return createUpdateCandidateInternally(*this, scoreVector);
            }
    };

}
