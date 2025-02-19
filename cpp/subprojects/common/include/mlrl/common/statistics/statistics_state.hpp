/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

/**
 * Defines an interface for all classes that allow to update statistics during the training process.
 *
 * @tparam ScoreType The type of the scores that are used for updating statistics
 */
template<typename ScoreType>
class IStatisticsState {
    public:

        virtual ~IStatisticsState() {}

        /**
         * The type of the scores that are used for updating statistics.
         */
        typedef ScoreType score_type;

        /**
         * Adds given scores to the predictions for all available outputs and updates affected statistics at a specific
         * index.
         *
         * @param statisticIndex    The index of the statistics to be updated
         * @param scoresBegin       An iterator to the beginning of the scores to be added
         * @param scoresEnd         An iterator to the end of the scores to be added
         * @param indicesBegin      An iterator to the beginning of the output indices
         * @param indicesEnd        An iterator to the end of the output indices
         */
        virtual void update(uint32 statisticIndex, typename View<ScoreType>::const_iterator scoresBegin,
                            typename View<ScoreType>::const_iterator scoresEnd,
                            CompleteIndexVector::const_iterator indicesBegin,
                            CompleteIndexVector::const_iterator indicesEnd) = 0;

        /**
         * Adds given scores to the predictions for a subset of the available outputs and updates affected statistics at
         * a specific index.
         *
         * @param statisticIndex    The index of the statistics to be updated
         * @param scoresBegin       An iterator to the beginning of the scores to be added
         * @param scoresEnd         An iterator to the end of the scores to be added
         * @param indicesBegin      An iterator to the beginning of the output indices
         * @param indicesEnd        An iterator to the end of the output indices
         */
        virtual void update(uint32 statisticIndex, typename View<ScoreType>::const_iterator scoresBegin,
                            typename View<ScoreType>::const_iterator scoresEnd,
                            PartialIndexVector::const_iterator indicesBegin,
                            PartialIndexVector::const_iterator indicesEnd) = 0;

        /**
         * Removes given scores from the predictions for all available outputs and updates affected statistics at a
         * specific index.
         *
         * @param statisticIndex    The index of the statistics to be updated
         * @param scoresBegin       An iterator to the beginning of the scores to be removed
         * @param scoresEnd         An iterator to the end of the scores to be removed
         * @param indicesBegin      An iterator to the beginning of the output indices
         * @param indicesEnd        An iterator to the end of the output indices
         */
        virtual void revert(uint32 statisticIndex, typename View<ScoreType>::const_iterator scoresBegin,
                            typename View<ScoreType>::const_iterator scoresEnd,
                            CompleteIndexVector::const_iterator indicesBegin,
                            CompleteIndexVector::const_iterator indicesEnd) = 0;

        /**
         * Removes given scores from the predictions for a subset of the available outputs and updates affected
         * statistics at a specific index.
         *
         * @param statisticIndex    The index of the statistics to be updated
         * @param scoresBegin       An iterator to the beginning of the scores to be removed
         * @param scoresEnd         An iterator to the end of the scores to be removed
         * @param indicesBegin      An iterator to the beginning of the output indices
         * @param indicesEnd        An iterator to the end of the output indices
         */
        virtual void revert(uint32 statisticIndex, typename View<ScoreType>::const_iterator scoresBegin,
                            typename View<ScoreType>::const_iterator scoresEnd,
                            PartialIndexVector::const_iterator indicesBegin,
                            PartialIndexVector::const_iterator indicesEnd) = 0;
};
