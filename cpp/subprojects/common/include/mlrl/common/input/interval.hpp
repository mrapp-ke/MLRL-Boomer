/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/tuple.hpp"

/**
 * Specifies the boundaries of an interval that includes/excludes certain elements in a vector.
 */
struct Interval {
    public:

        /**
         * The index of the first element to be included.
         */
        uint32 start;

        /**
         * The index of the last element to be included (exclusive).
         */
        uint32 end;

        /**
         * True, if the elements in the range [start, end) should be excluded, rather than included, false otherwise.
         */
        bool inverse;

        Interval() {}

        /**
         * @param start     The index of the first element to be included
         * @param end       The index of the last elements to be included (exclusive)
         * @param inverse   True, if the elements in the range [start, end) should be excluded, rather than included,
         *                  false otherwise
         */
        Interval(uint32 start, uint32 end, bool inverse = false) : start(start), end(end), inverse(inverse) {}

        virtual ~Interval() {}

        /**
         * Assigns the properties of an existing interval to this interval.
         *
         * @param rhs   A reference to the existing interval
         * @return      A reference to the modified interval
         */
        Interval& operator=(const Interval& rhs) {
            start = rhs.start;
            end = rhs.end;
            inverse = rhs.inverse;
            return *this;
        }
};

/**
 * Returns the start and end index of an open interval `[0, interval.end]` or `[interval.start, maxIndex]`, depending on
 * a given `Interval`.
 *
 * @param interval  A reference to an object of type `Interval`
 * @param maxIndex  The maximum index of an open interval
 * @return          A `Tuple` that stores the start and end index
 */
static inline Tuple<uint32> getStartAndEndOfOpenInterval(const Interval& interval, uint32 maxIndex) {
    Tuple<uint32> tuple;

    if (interval.inverse) {
        if (interval.start > 0) {
            tuple.first = 0;
            tuple.second = interval.start;
        } else {
            tuple.first = interval.end;
            tuple.second = maxIndex;
        }
    } else {
        tuple.first = interval.start;

        if (tuple.first > 0) {
            tuple.second = maxIndex;
        } else {
            tuple.second = interval.end;
        }
    }

    return tuple;
}
