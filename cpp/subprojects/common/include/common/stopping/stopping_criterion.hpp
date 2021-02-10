/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/partition.hpp"
#include "common/statistics/statistics.hpp"


/**
 * Defines an interface for all stopping criteria that allow to decide whether additional rules should be induced or
 * not.
 */
class IStoppingCriterion {

    public:

        virtual ~IStoppingCriterion() { };

        /**
         * Returns whether additional rules should be induced or not.
         *
         * @param partition     A reference to an object of type `IPartition` that provides access to the indices of the
         *                      training examples that belong to the training set and the holdout set, respectively
         * @param statistics    A reference to an object of type `IStatistics` that will serve as the basis for learning
         *                      the next rule
         * @param numRules      The number of rules induced so far
         * @return              True, if additional rules should be induced, false otherwise
         */
        virtual bool shouldContinue(const IPartition& partition, const IStatistics& statistics, uint32 numRules) = 0;

};
