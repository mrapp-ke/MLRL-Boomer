/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/stopping/stopping_criterion.hpp"

#include <memory>

/**
 * Allows to configure a stopping criterion that does not actually stop the induction of rules.
 */
class NoStoppingCriterionConfig final : public IStoppingCriterionConfig {
    public:

        std::unique_ptr<IStoppingCriterionFactory> createStoppingCriterionFactory() const override;
};
