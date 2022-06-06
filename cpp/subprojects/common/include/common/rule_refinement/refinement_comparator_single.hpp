/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_refinement/refinement.hpp"
#include "common/rule_refinement/score_processor.hpp"


/**
 * Allows comparing potential refinements of a rule and keeping track of the best one.
 */
class SingleRefinementComparator final {

    private:

        std::unique_ptr<Refinement> bestRefinementPtr_;

        float64 bestQualityScore_;

        ScoreProcessor scoreProcessor_;

    public:

        /**
         * @param bestHead A pointer to an object of type `AbstractEvaluatedPrediction`, representing the head of the
         *                 best rule found so far, or a null pointer, if no such rule is available
         */
        SingleRefinementComparator(const AbstractEvaluatedPrediction* bestHead);

        /**
         * Returns whether the quality of a rule's predictions is considered as an improvement over the quality of the
         * best rule found so far or not.
         *
         * @param scoreVector   A reference to an object of type `IScoreVector` that stores the quality of the
         *                      predictions
         * @return              True, if the quality of the given predictions is considered as an improvement, false
         *                      otherwise
         */
        bool isImprovement(const IScoreVector& scoreVector) const;

        /**
         * Keeps track of a specific refinement of a rule that is currently considered as the best one.
         *
         * @param refinement    A reference to an object of type `Refinement` that represents the refinement of the rule
         * @param scoreVector   A reference to an object of type `IScoreVector` that stores the predictions of the rule
         */
        void pushRefinement(const Refinement& refinement, const IScoreVector& scoreVector);

        /**
         * Returns the best refinement that has been passed to the function `pushRefinement`.
         *
         * @return An unique pointer to an object of type `Refinement` that stores information about the best refinement
         *         that has been found
         */
        std::unique_ptr<Refinement> pollRefinement();

};
