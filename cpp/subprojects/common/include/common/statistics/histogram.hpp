/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_immutable.hpp"


/**
 * Defines an interface for all classes that provide access to statistics that are organized as a histogram, i.e., where
 * the statistics of multiple training examples are aggregated into the same bin.
 */
class IHistogram : virtual public IImmutableStatistics {

    public:

        virtual ~IHistogram() { };

         /**
          * Removes the statistic at a specific index from a specific bin.
          *
          * @param binIndex         The index of the bin
          * @param statisticIndex   The index of the statistic
          * @param weight           The weight of the statistic
          */
        virtual void removeFromBin(uint32 binIndex, uint32 statisticIndex, float64 weight) = 0;

};
