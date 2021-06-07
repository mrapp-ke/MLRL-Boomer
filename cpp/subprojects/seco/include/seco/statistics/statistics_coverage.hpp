/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics.hpp"
#include "seco/data/matrix_dense_weights.hpp"


namespace seco {

    /**
     * Defines an interface for all classes that allow to store the elements of confusion matrices that have been
     * computed based on a weight matrix and the ground truth labels of the training examples.
     */
    class ICoverageStatistics : public IStatistics {

        public:

            virtual ~ICoverageStatistics() { };

            /**
             * Returns the sum of the weights of all labels that remain to be covered.
             *
             * @return The sum of the weights
             */
            virtual float64 getSumOfUncoveredWeights() const = 0;

            /**
             * Returns the weights of all labels that remains to be covered.
             *
             * @return The weights
             */
            virtual DenseWeightMatrix* getUncoveredWeights() const = 0;

    };

}
