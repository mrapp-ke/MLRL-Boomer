/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "statistics_immutable.h"


/**
 * Defines an interface for all classes that provide access to statistics that are organized as a histogram, i.e., where
 * the statistics of multiple training examples are aggregated into the same bin.
 */
class IHistogram : virtual public IImmutableStatistics {

    public:

        virtual ~IHistogram() { };

         /**
          * Removes the statistics at a specific index from a specific bin.
          *
          * @param binIndex          The index of the bin
          * @param statisticIndex    The index of the statistics
          */
        virtual void removeFromBin(uint32 binIndex, uint32 statisticIndex) = 0;

};
