/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_refinement/refinement_comparator_single.hpp"


/**
 * Defines an interface for all classes that allow to find the best refinement of existing rules.
 */
class IRuleRefinement {

    public:

        virtual ~IRuleRefinement() { };

        /**
         * Finds the best refinement of an existing rule.
         *
         * @param comparator A reference to an object of type `SingleRefinementComparator` that is used to compare the
         *                   potential refinements
         */
        virtual void findRefinement(SingleRefinementComparator& comparator) = 0;

};
