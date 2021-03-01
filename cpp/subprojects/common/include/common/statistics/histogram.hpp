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

};
