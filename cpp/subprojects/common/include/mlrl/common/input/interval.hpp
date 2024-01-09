/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

/**
 * Specifies the boundaries of an interval that includes/excludes certain elements in a vector.
 */
struct Interval final {
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

        /**
         * @param start     The index of the first element to be included
         * @param end       The index of the last elements to be included (exclusive)
         * @param inverse   True, if the elements in the range [start, end) should be excluded, rather than included,
         *                  false otherwise
         */
        Interval(uint32 start, uint32 end, bool inverse = false) : start(start), end(end), inverse(inverse) {}
};
