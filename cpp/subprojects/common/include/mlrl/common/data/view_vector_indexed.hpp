/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"

/**
 * A vector that is backed two one-dimensional views of a specific size, storing indices and corresponding values.
 *
 * @tparam IndexView    The type of the view, the indices are backed by
 * @tparam ValueView    The type of the view, the values are backed by
 */
template<typename IndexView, typename ValueView>
class IndexedVectorDecorator {
    protected:

        /**
         * The view, the indices are backed by.
         */
        IndexView indexView_;

        /**
         * The view, the values are backed by.
         */
        ValueView valueView_;

        /**
         * The type of the view, the indices are backed by.
         */
        typedef IndexView index_view_type;

        /**
         * The type of the view, the values are backed by.
         */
        typedef ValueView value_view_type;

    public:

        /**
         * @param indexView The view, the indices should be backed by
         * @param valueView The view, the values should be backed by
         */
        IndexedVectorDecorator(IndexView&& indexView, ValueView&& valueView)
            : indexView_(std::move(indexView)), valueView_(std::move(valueView)) {}

        virtual ~IndexedVectorDecorator() {};

        /**
         * The type of the indices that are stored in the vector.
         */
        typedef typename IndexView::value_type index_type;

        /**
         * The type of the values that are stored in the vector.
         *
         */
        typedef typename ValueView::value_type value_type;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const {
            return indexView_.numElements;
        }
};

/**
 * Provides read-only access via iterators to indices and corresponding values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class ReadIterableIndexedVectorDecorator : public Vector {
    public:

        /**
         * @param indexView The view, the indices should be backed by
         * @param valueView The view, the values should be backed by
         */
        ReadIterableIndexedVectorDecorator(typename Vector::index_view_type&& indexView,
                                           typename Vector::value_view_type&& valueView)
            : Vector(std::move(indexView), std::move(valueView)) {}

        virtual ~ReadIterableIndexedVectorDecorator() override {};

        /**
         * An iterator that provides read-only access to the indices stored in the vector.
         */
        typedef typename Vector::index_view_type::const_iterator index_const_iterator;

        /**
         * An iterator that provides read-only access to the values stored in the vector.
         */
        typedef typename Vector::value_view_type::const_iterator value_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the vector.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const {
            return Vector::indexView_.array;
        }

        /**
         * Returns an `index_const_iterator` to the end of the vector.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const {
            return &Vector::indexView_.array[Vector::indexView_.numElements];
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the vector.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const {
            return Vector::valueView_.array;
        }

        /**
         * Returns a `value_const_iterator` to the end of the vector.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const {
            return &Vector::valueView_.array[Vector::valueView_.numElements];
        }
};

/**
 * Provides write access via iterators to the indices and corresponding values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class WriteIterableIndexedVectorDecorator : public Vector {
    public:

        /**
         * @param indexView The view, the indices should be backed by
         * @param valueView The view, the values should be backed by
         */
        WriteIterableIndexedVectorDecorator(typename Vector::index_view_type&& indexView,
                                            typename Vector::value_view_type&& valueView)
            : Vector(std::move(indexView), std::move(valueView)) {}

        virtual ~WriteIterableIndexedVectorDecorator() override {};

        /**
         * An iterator that provides access to the indices stored in the vector and allows to modify them.
         */
        typedef typename Vector::index_view_type::iterator index_iterator;

        /**
         * An iterator that provides access to the values stored in the vector and allows to modify them.
         */
        typedef typename Vector::value_view_type::iterator value_iterator;

        /**
         * Returns an `index_iterator` to the beginning of the vector.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin() const {
            return Vector::indexView_.array;
        }

        /**
         * Returns an `index_iterator` to the end of the vector.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end() const {
            return &Vector::indexView_.array[Vector::indexView_.numElements];
        }

        /**
         * Returns a `value_iterator` to the beginning of the vector.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin() const {
            return Vector::valueView_.array;
        }

        /**
         * Returns a `value_iterator` to the end of the vector.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end() const {
            return &Vector::valueView_.array[Vector::valueView_.numElements];
        }
};

/**
 * Provides read-only access via iterators to indices and corresponding values stored in a vector.
 *
 * @tparam IndexVector  The type of the view, the indices are backed by
 * @tparam ValueVector  The type of the view, the values are backed by
 */
template<typename IndexView, typename ValueView>
using ReadableIndexedVectorDecorator = ReadIterableIndexedVectorDecorator<IndexedVectorDecorator<IndexView, ValueView>>;

/**
 * Provides read and write access via iterators to indices and corresponding values stored in a vector.
 *
 * @tparam IndexVector  The type of the view, the indices are backed by
 * @tparam ValueVector  The type of the view, the values are backed by
 */
template<typename IndexView, typename ValueView>
using WritableIndexedVectorDecorator =
  WriteIterableIndexedVectorDecorator<ReadableIndexedVectorDecorator<IndexView, ValueView>>;
