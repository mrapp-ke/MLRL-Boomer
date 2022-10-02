/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <functional>


/**
 * A struct that stores a numerical score that represents a quality.
 */
struct Quality {

    /**
     * A function for comparing two objects of type `Quality`. It should return true, if the first object is better than
     * the second one, false otherwise.
     */
    typedef std::function<bool(const Quality&, const Quality&)> CompareFunction;

    Quality() { };

    /**
     * @param q A numerical score that represents the quality
     */
    Quality(float64 q) : quality(q) { };

    /**
     * @param q A reference to an object of type `Quality` to be copied
     */
    Quality(const Quality& q) : quality(q.quality) { };

    /**
     * Assigns the numerical score of an existing object to this object.
     *
     * @param q A reference to the existing object
     * @return  A reference to the modified object
     */
    Quality& operator=(const Quality& q) {
        quality = q.quality;
        return *this;
    }

    /**
     * A numerical score that represents the quality.
     */
    float64 quality;

};

/**
 * Returns whether a specific object of type `Quality` is better than a second one or not.
 *
 * @param compareFunction   A function of type `QualityCompareFunction` that should be used for comparing the objects
 * @param first             A reference to the first object of type `Quality`
 * @param second            A reference to the second object of type `Quality`
 * @return                  True, if the first object is better than the second one, false otherwise
 */
static inline bool compareQuality(Quality::CompareFunction compareFunction, const Quality& first,
                                  const Quality& second) {
    return compareFunction(first, second);
}
