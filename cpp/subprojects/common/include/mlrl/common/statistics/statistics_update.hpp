/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow updating statistics.
 */
class IStatisticsUpdate {
    public:

        virtual ~IStatisticsUpdate() {}

        /**
         * Updates a specific statistic.
         *
         * This function must be called for each statistic that is covered by a new rule before learning the
         * next rule.
         *
         * @param statisticIndex The index of the statistic that should be updated
         */
        virtual void applyPrediction(uint32 statisticIndex) = 0;

        /**
         * Reverts a specific statistic that has previously been updated via the function `applyPrediction`.
         *
         * @param statisticIndex The index of the statistic that should be updated
         */
        virtual void revertPrediction(uint32 statisticIndex) = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IStatisticsUpdate`.
 *
 * @tparam ScoreType The type of the scores that are used for updating statistics
 */
template<typename ScoreType>
class IStatisticsUpdateFactory {
    public:

        virtual ~IStatisticsUpdateFactory() {}

        /**
         * Creates and returns a new instance of type `IStatisticsUpdate` for updating statistics for all of the
         * available outputs.
         *
         * @param indicesBegin  An iterator to the beginning of the output indices for which statistics should be
         *                      updated
         * @param indicesEnd    An iterator to the end of the output indices for which statistics should be updated
         * @param scoresBegin   An iterator to the beginning of the predicted scores, corresponding to the given output
         *                      indices, that should be used for updating the statistics
         * @param scoresEnd     An iterator to the end of the predicted scores, corresponding to the given output
         *                      indices, that should be used for updating the statistics
         * @return              An unique pointer to an object of type `IStatisticsUpdate` that has been created
         */
        virtual std::unique_ptr<IStatisticsUpdate> create(CompleteIndexVector::const_iterator indicesBegin,
                                                          CompleteIndexVector::const_iterator indicesEnd,
                                                          typename View<ScoreType>::const_iterator scoresBegin,
                                                          typename View<ScoreType>::const_iterator scoresEnd) = 0;

        /**
         * Creates and returns a new instance of type `IStatisticsUpdate` for updating statistics for a subset of the
         * available outputs.
         *
         * @param indicesBegin  An iterator to the beginning of the output indices for which statistics should be
         *                      updated
         * @param indicesEnd    An iterator to the end of the output indices for which statistics should be updated
         * @param scoresBegin   An iterator to the beginning of the predicted scores, corresponding to the given output
         *                      indices, that should be used for updating the statistics
         * @param scoresEnd     An iterator to the end of the predicted scores, corresponding to the given output
         *                      indices, that should be used for updating the statistics
         * @return              An unique pointer to an object of type `IStatisticsUpdate` that has been created
         */
        virtual std::unique_ptr<IStatisticsUpdate> create(PartialIndexVector::const_iterator indicesBegin,
                                                          PartialIndexVector::const_iterator indicesEnd,
                                                          typename View<ScoreType>::const_iterator scoresBegin,
                                                          typename View<ScoreType>::const_iterator scoresEnd) = 0;
};
