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

        Refinement bestRefinement_;

        float64 bestQualityScore_;

        ScoreProcessor scoreProcessor_;

    public:

        SingleRefinementComparator();

        /**
         * @param comparator A reference to an object of type `SingleRefinementComparator` that keeps track of the best
         *                   refinement found so far
         */
        SingleRefinementComparator(const SingleRefinementComparator& comparator);

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
         * Keeps track of a given refinement of a rule that is considered as an improvement over the best rule found so
         * far.
         *
         * @param refinement    A reference to an object of type `Refinement` that represents the refinement of the rule
         * @param scoreVector   A reference to an object of type `IScoreVector` that stores the predictions of the rule
         */
        void pushRefinement(const Refinement& refinement, const IScoreVector& scoreVector);

        /**
         * Keeps track of the best refinement that is stored by a given `SingleRefinementComparator` if it is considered
         * as an improvement over the best refinement that has been provided to this comparator.
         *
         * @param comparator    A reference to an object of type `SingleRefinementComparator` that should be merged
         * @return              True, if the best refinement that is stored by the given `comparator` is considered as
         *                      an improvement over the best refinement that has been provided to this comparator
         */
        bool merge(SingleRefinementComparator& comparator);

        /**
         * Returns the best refinement that has been provided via the function `pushRefinement` or `merge`.
         *
         * @return A reference to an object of type `Refinement` that represents the best refinement that has been
         *         provided
         */
        Refinement& getBestRefinement();

};
