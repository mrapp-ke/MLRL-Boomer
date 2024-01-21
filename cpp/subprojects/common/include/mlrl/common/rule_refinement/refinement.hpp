/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/model/condition.hpp"
#include "mlrl/common/rule_refinement/prediction_evaluated.hpp"

/**
 * Stores the properties of a potential refinement of a rule.
 */
struct Refinement final : public Condition {
    public:

        /**
         * Assigns the properties of an existing refinement, except for the scores that are predicted by the refined
         * rule, to this refinement.
         *
         * @param rhs   A reference to the existing refinement
         * @return      A reference to the modified refinement
         */
        Refinement& operator=(const Refinement& rhs) {
            Condition::operator=(rhs);
            return *this;
        }

        /**
         * An unique pointer to an object of type `IEvaluatedPrediction` that stores the scores that are predicted by
         * the refined rule, as well as its overall quality.
         */
        std::unique_ptr<IEvaluatedPrediction> headPtr;
};
