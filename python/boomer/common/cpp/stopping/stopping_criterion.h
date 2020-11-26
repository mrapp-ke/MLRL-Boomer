/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../statistics/statistics.h"


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
         * @param statistics    A reference to an object of type `IStatistics`, which will serve as the basis for
         *                      learning the next rule
         * @param numRules      The number of rules induced so far
         * @return              True, if additional rules should be induced, false otherwise
         */
        virtual bool shouldContinue(const IStatistics& statistics, uint32 numRules) = 0;

};
