/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_composite.hpp"
#include "mlrl/common/data/view_vector.hpp"

/**
 * Allows to set all values stored in a vector that is backed by two one-dimensional views to zero.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class MLRLCOMMON_API ClearableCompositeVectorDecorator : public Vector {
    public:

        /**
         * @param firstView     The first view, the vector should be backed by
         * @param secondView    The second view, the vector should be backed by
         */
        ClearableCompositeVectorDecorator(typename Vector::first_view_type&& firstView,
                                          typename Vector::second_view_type&& secondView)
            : Vector(std::move(firstView), std::move(secondView)) {}

        /**
         * Sets all values stored in the vector to zero.
         */
        virtual void clear() {
            setViewToZeros(Vector::firstView_.array, Vector::firstView_.numElements);
            setViewToZeros(Vector::secondView_.array, Vector::secondView_.numElements);
        }
};
