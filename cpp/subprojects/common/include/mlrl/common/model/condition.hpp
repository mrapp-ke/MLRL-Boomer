/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

/**
 * An enum that specifies all possible types of operators used by a condition of a rule.
 */
enum Comparator : uint8 {
    NUMERICAL_LEQ = 0,
    NUMERICAL_GR = 1,
    ORDINAL_LEQ = 2,
    ORDINAL_GR = 3,
    NOMINAL_EQ = 4,
    NOMINAL_NEQ = 5
};

/**
 * A union of types that may be used for the threshold used by a condition of a rule.
 */
union Threshold {
        float32 numerical;
        int32 nominal;
        int32 ordinal;
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
         * @param other A reference to an existing condition to be copied
         */
        Condition(const Condition& other)
            : featureIndex(other.featureIndex), comparator(other.comparator), threshold(other.threshold),
              start(other.start), end(other.end), covered(other.covered), numCovered(other.numCovered) {}

        /**
         * Assigns the properties of an existing condition to this condition.
         *
         * @param rhs   A reference to the existing condition
         * @return      A reference to the modified condition
         */
        Condition& operator=(const Condition& rhs) {
            featureIndex = rhs.featureIndex;
            comparator = rhs.comparator;
            threshold = rhs.threshold;
            start = rhs.start;
            end = rhs.end;
            covered = rhs.covered;
            numCovered = rhs.numCovered;
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
        Threshold threshold;

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
        // TODO Remove, if possible
        uint32 numCovered;
};
