/**
 * Implements base classes for all classes that allow to store gradient statistics that are computed according to a
 * differentiable loss function based on the current predictions of rules and the ground truth labels of the training
 * examples.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/statistics.h"


namespace boosting {

    /**
     * An abstract base class for all classes that store gradient statistics.
     */
    class AbstractGradientStatistics : public IStatistics {

        private:

            uint32 numStatistics_;

            uint32 numLabels_;

        public:

            /**
             * @param numStatistics The number of statistics
             * @param numLabels     The number of labels
             */
            AbstractGradientStatistics(uint32 numStatistics, uint32 numLabels);

            uint32 getNumStatistics() const override;

            uint32 getNumLabels() const override;

            void resetSampledStatistics() override;

            void addSampledStatistic(uint32 statisticIndex, uint32 weight) override;

    };

}
