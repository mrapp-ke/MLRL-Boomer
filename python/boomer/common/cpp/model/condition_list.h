/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "condition.h"
#include <list>
#include <array>


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
