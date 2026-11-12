/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/stopping/stopping_criterion.hpp"
#include "mlrl/seco/util/dll_exports.hpp"

#include <memory>

namespace seco {

    /**
     * Defines an interface for all classes that allow to configure a stopping criterion that stops the induction of
     * rules as soon as a certain fraction of the available training examples and labels is covered.
     */
    class MLRLSECO_API ICoverageStoppingCriterionConfig {
        public:

            virtual ~ICoverageStoppingCriterionConfig() {}

            /**
             * Returns the fraction of training examples and labels that must be covered before the induction of rules
             * is stopped.
             *
             * @return The fraction that must be covered before the induction of rules is stopped
             */
            virtual float32 getMinCoverage() const = 0;

            /**
             * Sets the fraction of training examples and labels that must be covered before the induction of rules is
             * stopped.
             *
             * @param minCoverage   The fraction of training examples and labels that must be covered before the
             *                      induction of rules is stopped. Must be in [0, 1)
             * @return              A reference to an object of type `ICoverageStoppingCriterionConfig` that allows
             *                      further configuration of the stopping criterion
             */
            virtual ICoverageStoppingCriterionConfig& setMinCoverage(float32 minCoverage) = 0;
    };

    /**
     * Allows to configure a stopping criterion that stops the induction of rules as soon as a certain fraction of the
     * available training examples and labels is covered.
     */
    class CoverageStoppingCriterionConfig final : public IStoppingCriterionConfig,
                                                  public ICoverageStoppingCriterionConfig {
        private:

            float32 minCoverage_;

        public:

            CoverageStoppingCriterionConfig();

            float32 getMinCoverage() const override;

            ICoverageStoppingCriterionConfig& setMinCoverage(float32 minCoverage) override;

            /**
             * @see `IStoppingCriterionConfig::createStoppingCriterionFactory`
             */
            std::unique_ptr<IStoppingCriterionFactory> createStoppingCriterionFactory() const override;
    };

}
