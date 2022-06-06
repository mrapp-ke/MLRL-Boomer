/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/condition.hpp"
#include "common/rule_refinement/prediction_evaluated.hpp"


/**
 * Stores the properties of a potential refinement of a rule.
 */
struct Refinement : public Condition {

    /**
     * Returns whether this refinement is better than another one.
     *
     * @param another   A reference to an object of type `Refinement` to be compared to
     * @return          True, if this refinement is better than the given one, false otherwise
     */
    bool isBetterThan(const Refinement& another) const {
        const AbstractEvaluatedPrediction* head = headPtr.get();

        if (head) {
            const AbstractEvaluatedPrediction* anotherHead = another.headPtr.get();
            return !anotherHead || head->overallQualityScore < anotherHead->overallQualityScore;
        }

        return false;
    }

    /**
     * An unique pointer to an object of type `AbstractEvaluatedPrediction` that stores the scores that are
     * predicted by the refined rule, as well as a corresponding quality score.
     */
    std::unique_ptr<AbstractEvaluatedPrediction> headPtr;

    /**
     * The index of the last element, e.g., example or bin, that has been processed when evaluating the refined
     * rule.
     */
    int64 previous;

};
