/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_state.hpp"

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
     * @tparam LossFunction     The type of the loss function that is used to calculate gradients and Hessians
     */
    template<typename OutputMatrix, typename StatisticMatrix, typename ScoreMatrix, typename LossFunction>
    class AbstractStatisticsState : public IStatisticsState<typename ScoreMatrix::value_type> {
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
             * An unique pointer to an object of template type `LossFunction` that is used for calculating gradients and
             * Hessians.
             */
            std::unique_ptr<LossFunction> lossFunctionPtr;

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
             * @param lossFunctionPtr       An unique pointer to the an object of template type `LossFunction` that
             *                              should be used for calculating gradients and Hessians
             */
            AbstractStatisticsState(const OutputMatrix& outputMatrix,
                                    std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                                    std::unique_ptr<ScoreMatrix> scoreMatrixPtr,
                                    std::unique_ptr<LossFunction> lossFunctionPtr)
                : outputMatrix(outputMatrix), statisticMatrixPtr(std::move(statisticMatrixPtr)),
                  scoreMatrixPtr(std::move(scoreMatrixPtr)), lossFunctionPtr(std::move(lossFunctionPtr)) {}

            virtual ~AbstractStatisticsState() {}

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
    };

}
