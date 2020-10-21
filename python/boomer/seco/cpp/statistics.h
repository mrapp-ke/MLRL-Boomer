/**
 * Implements base classes for all classes that allow to store the elements of confusion matrices that are computed
 * based on a weight matrix and the ground truth labels of the training examples.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
# pragma once

#include "../../common/cpp/statistics.h"


namespace seco {

    /**
     * An abstract base class for all classes that allow to store the elements of confusion matrices that are computed
     * based on a weight matrix and the ground truth labels of the training examples.
     */
    class AbstractCoverageStatistics : public AbstractStatistics {

        public:

            /**
             * @param numStatistics         The number of statistics
             * @param numLabels             The number of labels
             * @param sumUncoveredLabels    The sum of weights of all labels that remain to be covered, initially
             */
            AbstractCoverageStatistics(uint32 numStatistics, uint32 numLabels, float64 sumUncoveredLabels);

            /**
             * The sum of weights of all labels that remain to be covered.
             */
            float64 sumUncoveredLabels_;

    };

}
