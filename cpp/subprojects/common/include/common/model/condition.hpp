/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

/**
 * An enum that specifies all possible types of operators used by a condition of a rule.
 */
enum Comparator : uint8 {
    LEQ = 0,
    GR = 1,
    EQ = 2,
    NEQ = 3
};

/**
 * Stores the properties of a condition of a rule. It consists of the index of the feature, the condition corresponds
 * to, the type of the operator that is used by the condition, as well as a threshold. In addition, it stores the range
 * [start, end) that corresponds to the elements, e.g. examples or bins, that are covered (or uncovered, if
 * `covered == false`) by the condition, as well as the sum of the weights of all covered elements.
 */
struct Condition {
    public:

        Condition() {}

        /**
         * @param condition A reference to an existing condition to be copied
         */
        Condition(const Condition& condition)
            : featureIndex(condition.featureIndex), comparator(condition.comparator), threshold(condition.threshold),
              start(condition.start), end(condition.end), covered(condition.covered), numCovered(condition.numCovered) {
        }

        /**
         * Assigns the properties of an existing condition to this condition.
         *
         * @param condition A reference to the existing condition
         * @return          A reference to the modified condition
         */
        Condition& operator=(const Condition& condition) {
            featureIndex = condition.featureIndex;
            comparator = condition.comparator;
            threshold = condition.threshold;
            start = condition.start;
            end = condition.end;
            covered = condition.covered;
            numCovered = condition.numCovered;
            return *this;
        }

        /**
         * The index of the feature, the condition corresponds to.
         */
        uint32 featureIndex;

        /**
         * The type of the operator that is used by the condition.
         */
        Comparator comparator;

        /**
         * The threshold that is used by the condition.
         */
        float32 threshold;

        /**
         * The index of the first element (inclusive) that is covered (or uncovered) by the condition.
         */
        int64 start;

        /**
         * The index of the last element (exclusive) that is covered (or uncovered) by the condition.
         */
        int64 end;

        /**
         * True, if the elements in [start, end) are covered by the condition, false otherwise.
         */
        bool covered;

        /**
         * The number of elements that are covered by the condition.
         */
        uint32 numCovered;
};
