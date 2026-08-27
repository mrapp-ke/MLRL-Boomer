/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics.hpp"

namespace seco {

    /**
     * Defines an interface for all classes that allow to store the elements of confusion matrices that are successively
     * covered.
     */
    class ICoverageStatistics : virtual public IStatistics {
        public:

            virtual ~ICoverageStatistics() override {}

            /**
             * Returns the fraction of statistics that remain to be covered.
             *
             * @return The fraction of statistics that remain to be covered
             */
            virtual float64 getUncoveredFraction() const = 0;
    };

}
