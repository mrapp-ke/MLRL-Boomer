/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_indexed.hpp"
#include "mlrl/common/iterator/binned_iterator.hpp"

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

/**
 *
 *
 * @tparam Vector
 */
/**
 * Provides read-only access via iterators to elements in a vector that are grouped into bins.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class ReadIterableBinnedVectorDecorator : public Vector {
    public:

        /**
         * @param binIndexView  The view, the bin indices should be backed by
         * @param valueView     The view, the values of the bins should be backed by
         */
        ReadIterableBinnedVectorDecorator(typename Vector::first_view_type&& binIndexView,
                                          typename Vector::second_view_type&& valueView)
            : Vector(std::move(binIndexView), std::move(valueView)) {}

        virtual ~ReadIterableBinnedVectorDecorator() override {};

        /**
         * An iterator that provides read-only access to the elements in the vector.
         */
        typedef BinnedConstIterator<typename Vector::value_type> const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return BinnedConstIterator<typename Vector::value_type>(this->firstView_.cbegin(),
                                                                    this->secondView_.cbegin());
        }

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const {
            return BinnedConstIterator<typename Vector::value_type>(this->firstView_.cend(),
                                                                    this->secondView_.cbegin());
        }
};

/**
 * Provides read-only access via iterators to elements in a vector that are grouped into bins, as well as to the bin
 * indices and the values of the corresponding values.
 *
 * @tparam BinIndexView The type of the view, the bin indices are backed by
 * @tparam ValueView    The type of the view, the values of the bins are backed by
 */
template<typename BinIndexView, typename ValueView>
using ReadableBinnedVectorDecorator =
  ReadIterableBinnedVectorDecorator<ReadIterableIndexedVectorDecorator<BinnedVectorDecorator<BinIndexView, ValueView>>>;

/**
 * Provides read and write access via iterators to elements in a vector that are grouped into bins, as well as to the
 * bin indices and the values of the corresponding bins.
 *
 * @tparam BinIndexView The type of the view, the bin indices are backed by
 * @tparam ValueView    The type of the view, the values of the bins are backed by
 */
template<typename BinIndexView, typename ValueView>
using WritableBinnedVectorDecorator =
  WriteIterableIndexedVectorDecorator<ReadableBinnedVectorDecorator<BinIndexView, ValueView>>;
