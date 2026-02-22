/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/interval.hpp"

#include <variant>

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
using Threshold = std::variant<float32, int32>;

/**
 * Stores the properties of a condition of a rule. It consists of the index of the feature, the condition corresponds
 * to, the type of the operator that is used by the condition, as well as a threshold. In addition, it stores the range
 * [start, end) that corresponds to the elements, e.g. examples or bins, that are covered (or uncovered, if
 * `covered == false`) by the condition, as well as the sum of the weights of all covered elements.
 */
struct Condition : public Interval {
    public:

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
         * The number of elements that are covered by the condition.
         */
        uint32 numCovered;

        Condition() : Interval() {}

        /**
         * @param other A reference to an existing condition to be copied
         */
        Condition(const Condition& other)
            : Interval(other.start, other.end, other.inverse), featureIndex(other.featureIndex),
              comparator(other.comparator), threshold(other.threshold), numCovered(other.numCovered) {}

        virtual ~Condition() override {}

        /**
         * Assigns the properties of an existing condition to this condition.
         *
         * @param rhs   A reference to the existing condition
         * @return      A reference to the modified condition
         */
        Condition& operator=(const Condition& rhs) {
            Interval::operator=(rhs);
            featureIndex = rhs.featureIndex;
            comparator = rhs.comparator;
            threshold = rhs.threshold;
            numCovered = rhs.numCovered;
            return *this;
        }
};
