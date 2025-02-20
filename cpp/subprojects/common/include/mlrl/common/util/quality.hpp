/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

/**
 * A struct that stores a numerical score that represents a quality.
 */
struct Quality {
    public:

        Quality() {}

        /**
         * @param quality A numerical score that represents the quality
         */
        explicit Quality(float64 quality) : quality(quality) {}

        /**
         * @param other A reference to an object of type `Quality` to be copied
         */
        Quality(const Quality& other) : quality(other.quality) {}

        virtual ~Quality() {}

        /**
         * Assigns the numerical score of an existing object to this object.
         *
         * @param rhs   A reference to the existing object
         * @return      A reference to the modified object
         */
        Quality& operator=(const Quality& rhs) {
            quality = rhs.quality;
            return *this;
        }

        /**
         * A numerical score that represents the quality.
         */
        float64 quality;
};
