/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics.hpp"

namespace seco {

    /**
     * Defines an interface for all classes that allow to store the elements of confusion matrices that have been
     * computed based on a weight matrix and the ground truth labels of the training examples.
     */
    class ICoverageStatistics : public IStatistics {
        public:

            virtual ~ICoverageStatistics() override {};

            /**
             * Returns the sum of the weights of all labels that remain to be covered.
             *
             * @return The sum of the weights
             */
            virtual float64 getSumOfUncoveredWeights() const = 0;
    };

}
