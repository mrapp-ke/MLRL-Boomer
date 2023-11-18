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
class BinnedVectorDecorator : public Vector {
    public:

        /**
         * @param binIndexView  The view, the bin indices should be backed by
         * @param valueView     The view, the values of the bins should be backed by
         */
        BinnedVectorDecorator(typename Vector::first_view_type&& binIndexView,
                              typename Vector::second_view_type&& valueView)
            : Vector(std::move(binIndexView), std::move(valueView)) {}

        virtual ~BinnedVectorDecorator() override {};

        /**
         * Returns the number of bins in the vector.
         *
         * @return The number of bins
         */
        uint32 getNumBins() const {
            return this->secondView_.numElements;
        }
};
