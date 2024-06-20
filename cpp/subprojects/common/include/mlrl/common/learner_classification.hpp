/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/learner.hpp"

#include <memory>
#include <utility>

/**
 * Defines an interface for all rule learners that can be applied to classification problems.
 */
class MLRLCOMMON_API IClassificationRuleLearner : virtual public IRuleLearner {
    public:

        virtual ~IClassificationRuleLearner() override {}
};
