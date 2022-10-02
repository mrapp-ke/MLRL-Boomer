/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/util/quality.hpp"


/**
 * Defines a function for comparing the quality of different rules.
 */
struct RuleCompareFunction {

    /**
     * @param f A function of type `Quality::CompareFunction` for comparing the quality of different rules
     */
    RuleCompareFunction(Quality::CompareFunction f) : function(f) { };

    /**
     * A function of type `Quality::CompareFunction` for comparing the quality of different rules.
     */
    Quality::CompareFunction function;

};
