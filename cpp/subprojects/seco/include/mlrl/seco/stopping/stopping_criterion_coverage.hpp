/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/stopping/stopping_criterion.hpp"
#include "mlrl/seco/macros.hpp"

namespace seco {

    /**
     * Defines an interface for all classes that allow to configure a stopping criterion that stops the induction of
     * rules as soon as the sum of the weights of the uncovered labels is smaller or equal to a certain threshold.
     */
    class MLRLSECO_API ICoverageStoppingCriterionConfig {
        public:

            virtual ~ICoverageStoppingCriterionConfig() {};

            /**
             * Returns the threshold that is used by the stopping criterion.
             *
             * @return The threshold that is used by the stopping criterion
             */
            virtual float64 getThreshold() const = 0;

            /**
             * Sets the threshold that should be used by the stopping criterion.
             *
             * @param threshold The threshold that should be used by the stopping criterion. The threshold must be at
             *                  least 0
             * @return          A reference to an object of type `ICoverageStoppingCriterionConfig` that allows further
             *                  configuration of the stopping criterion
             */
            virtual ICoverageStoppingCriterionConfig& setThreshold(float64 threshold) = 0;
    };

    /**
     * Allows to configure a stopping criterion that stops the induction of rules as soon as the sum of the weights of
     * the uncovered labels is smaller or equal to a certain threshold.
     */
    class CoverageStoppingCriterionConfig final : public IStoppingCriterionConfig,
                                                  public ICoverageStoppingCriterionConfig {
        private:

            float64 threshold_;

        public:

            CoverageStoppingCriterionConfig();

            float64 getThreshold() const override;

            ICoverageStoppingCriterionConfig& setThreshold(float64 threshold) override;

            std::unique_ptr<IStoppingCriterionFactory> createStoppingCriterionFactory() const override;
    };

}
