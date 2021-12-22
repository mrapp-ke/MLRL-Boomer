/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"


namespace seco {

    /**
     * A stopping criterion that stops when the sum of the weights of the uncovered labels provided by
     * `ICoverageStatistics` is smaller than or equal to a certain threshold.
     */
    class CoverageStoppingCriterion final : public IStoppingCriterion {

        private:

            float64 threshold_;

        public:

            /**
             * @param threshold The threshold. Must be at least 0
             */
            CoverageStoppingCriterion(float64 threshold);

            Result test(const IPartition& partition, const IStatistics& statistics, uint32 numRules) override;

    };

    /**
     * Allows to create instances of the type `IStoppingCriterion` that stop the induction of rules as soon as the sum
     * of the weights of the uncovered labels, as provided by an object of type `ICoverageStatistics`, is smaller than
     * or equal to a certain threshold.
     */
    class CoverageStoppingCriterionFactory final : public IStoppingCriterionFactory {

        private:

            float64 threshold_;

        public:

            /**
             * @param threshold The threshold. Must be at least 0
             */
            CoverageStoppingCriterionFactory(float64 threshold);

            std::unique_ptr<IStoppingCriterion> create() const override;

    };

}
