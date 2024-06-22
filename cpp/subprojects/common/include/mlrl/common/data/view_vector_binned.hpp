/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_indexed.hpp"

/**
 * A vector that is backed by two one-dimensional views, storing bin indices and the values of the corresponding bins.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class MLRLCOMMON_API BinnedVectorDecorator : public Vector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        explicit BinnedVectorDecorator(typename Vector::view_type&& view) : Vector(std::move(view)) {}

        virtual ~BinnedVectorDecorator() override {}

        /**
         * Returns the number of bins in the vector.
         *
         * @return The number of bins
         */
        uint32 getNumBins() const {
            return this->view.secondView.numElements;
        }
};
