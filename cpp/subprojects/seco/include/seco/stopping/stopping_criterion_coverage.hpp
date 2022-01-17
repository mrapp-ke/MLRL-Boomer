/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"


namespace seco {

    /**
     * Allows to configure a stopping criterion that stops the induction of rules as soon as the sum of the weights of
     * the uncovered labels is smaller or equal to a certain threshold.
     */
    class CoverageStoppingCriterionConfig : public IStoppingCriterionConfig {

        private:

            float64 threshold_;

        public:

            CoverageStoppingCriterionConfig();

            /**
             * Returns the threshold that is used by the stopping criterion.
             *
             * @return The threshold that is used by the stopping criterion
             */
            float64 getThreshold() const;

            /**
             * Sets the threshold that should be used by the stopping criterion.
             *
             * @param threshold The threshold that should be used by the stopping criterion. The threshold must be at
             *                  least 0
             * @return          A reference to an object of type `CoverageStoppingCriterionConfig` that allows further
             *                  configuration of the stopping criterion
             */
            CoverageStoppingCriterionConfig& setThreshold(float64 threshold);

    };

    /**
     * Allows to create instances of the type `IStoppingCriterion` that stop the induction of rules as soon as the sum
     * of the weights of the uncovered labels, as provided by an object of type `ICoverageStatistics`, is smaller or
     * equal to a certain threshold.
     */
    class CoverageStoppingCriterionFactory final : public IStoppingCriterionFactory {

        private:

            float64 threshold_;

        public:

            /**
             * @param threshold The threshold. Must be at least 0
             */
            CoverageStoppingCriterionFactory(float64 threshold);

            std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override;

            std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override;

    };

}
