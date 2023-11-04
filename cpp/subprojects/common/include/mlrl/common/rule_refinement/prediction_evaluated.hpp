/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_refinement/prediction.hpp"
#include "mlrl/common/util/quality.hpp"

/**
 * Defines an interface for all classes that store the scores that are predicted by a rule, as well as a numerical score
 * that assesses the overall quality of the rule.
 */
class IEvaluatedPrediction : public IPrediction,
                             public Quality {
    public:

        virtual ~IEvaluatedPrediction() override {};
};
