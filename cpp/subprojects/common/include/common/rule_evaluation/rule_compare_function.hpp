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
     * @param m The minimum quality of a rule
     */
    RuleCompareFunction(Quality::CompareFunction f, float64 m) : function(f), minQuality(m) { };

    /**
     * A function of type `Quality::CompareFunction` for comparing the quality of different rules.
     */
    Quality::CompareFunction function;

    /**
     * The minimum quality of a rule.
     */
    float64 minQuality;

};
