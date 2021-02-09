/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/model/condition.hpp"
#include "common/head_refinement/prediction_evaluated.hpp"


/**
 * Stores information about a potential refinement of a rule.
 */
class Refinement final : public Condition {

    public:

        /**
         * Returns whether this refinement is better than another one.
         *
         * @param   A reference to an object of type `Refinement` to be compared to
         * @return  True, if this refinement is better than the given one, false otherwise
         */
        bool isBetterThan(const Refinement& another) const;

        std::unique_ptr<AbstractEvaluatedPrediction> headPtr;

        intp previous;

};