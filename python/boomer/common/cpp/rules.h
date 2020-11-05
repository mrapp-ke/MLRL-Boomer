/**
 * Provides classes that are used to build rule-based models.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include <list>
#include <array>


/**
 * An enum that specifies all possible types of operators used by a condition of a rule.
 */
enum Comparator : uint32 {
    LEQ = 0,
    GR = 1,
    EQ = 2,
    NEQ = 3
};

/**
 * Stores information about a condition of a rule. It consists of the index of the feature, the condition corresponds
 * to, the type of the operator that is used by the condition, as well as a threshold. In addition, it stores the range
 * [start, end) that corresponds to the elements, e.g. examples or bins, that are covered (or uncovered, if
 * `covered == false`) by the condition, as well as the sum of the weights of all covered elements.
 */
struct Condition {
    uint32 featureIndex;
    Comparator comparator;
    float32 threshold;
    intp start;
    intp end;
    bool covered;
    uint32 coveredWeights;
};

/**
 * A list that stores conditions in the order they have been learned.
 */
class ConditionList {

    private:

        std::list<Condition> list_;

        std::array<uint32, 4> numConditionsPerComparator_ = {0, 0, 0, 0};

    public:

        typedef std::list<Condition>::size_type size_type;

        typedef std::list<Condition>::const_iterator const_iterator;

        const_iterator cbegin() const;

        const_iterator cend() const;

        /**
         * Returns how many conditions are contained by the list in total.
         *
         * @return The number of conditions that are contained by the list
         */
        size_type getNumConditions() const;

        /**
         * Returns how many conditions with a specific comparator are contained by the list.
         *
         * @param comparator The comparator
         * @return           The number of conditions with the given comparator that are contained by the list
         */
        uint32 getNumConditions(Comparator comparator) const;

        /**
         * Adds a new condition to the end of the list.
         *
         * @param condition The condition to be added
         */
        void append(Condition condition);

        /**
         * Removes the last condition from the list.
         */
        void removeLast();

};
