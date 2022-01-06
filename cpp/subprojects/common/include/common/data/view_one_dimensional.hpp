/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Defines an interface for all one-dimensional views.
 *
 * @tparam T The type of the values
 */
class IOneDimensionalView {

    public:

        virtual ~IOneDimensionalView() { };

        /**
         * Returns the number of elements in the view.
         *
         * @return The number of elements
         */
        virtual uint32 getNumElements() const = 0;

};
