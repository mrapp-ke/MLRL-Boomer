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
     * Allows to configure a stopping criterion that stops the induction of rules as soon as the entire label space is
     * covered.
     */
    class CoverageStoppingCriterionConfig final : public IStoppingCriterionConfig {
        public:

            /**
             * @see `IStoppingCriterionConfig::createStoppingCriterionFactory`
             */
            std::unique_ptr<IStoppingCriterionFactory> createStoppingCriterionFactory() const override;
    };

}
