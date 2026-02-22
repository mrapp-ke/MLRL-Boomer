/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/util/quality.hpp"

#include <functional>

/**
 * Defines a function for comparing the quality of different rules.
 */
struct RuleCompareFunction {
    public:

        /**
         * A function for comparing two objects of type `Quality`. It should return true, if the first object is better
         * than the second one, false otherwise.
         */
        using CompareFunction = std::function<bool(const Quality&, const Quality&)>;

        /**
         * @param compareFunction   A function of type `CompareFunction` for comparing the quality of different rules
         * @param minQuality        The minimum quality of a rule
         */
        RuleCompareFunction(CompareFunction compareFunction, float64 minQuality)
            : compare(compareFunction), minQuality(minQuality) {}

        /**
         * A function of type `CompareFunction` for comparing the quality of different rules.
         */
        const CompareFunction compare;

        /**
         * The minimum quality of a rule.
         */
        const float64 minQuality;
};
