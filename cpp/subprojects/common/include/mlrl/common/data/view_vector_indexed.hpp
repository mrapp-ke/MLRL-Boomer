/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_composite.hpp"

/**
 * A vector that is backed two one-dimensional views of a specific size, storing indices and corresponding values.
 *
 * @tparam IndexView    The type of the view, the indices are backed by
 * @tparam ValueView    The type of the view, the values are backed by
 */
template<typename IndexView, typename ValueView>
class IndexedVectorDecorator : public CompositeVectorDecorator<IndexView, ValueView> {
    public:

        /**
         * @param indexView The view, the indices should be backed by
         * @param valueView The view, the values should be backed by
         */
        IndexedVectorDecorator(IndexView&& indexView, ValueView&& valueView)
            : CompositeVectorDecorator<IndexView, ValueView>(std::move(indexView), std::move(valueView)) {}

        virtual ~IndexedVectorDecorator() override {};

        /**
         * The type of the indices that are stored in the vector.
         */
        typedef typename IndexView::value_type index_type;

        /**
         * The type of the values that are stored in the vector.
         */
        typedef typename ValueView::value_type value_type;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const {
            return this->firstView_.numElements;
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
        ReadIterableIndexedVectorDecorator(typename Vector::first_view_type&& indexView,
                                           typename Vector::second_view_type&& valueView)
            : Vector(std::move(indexView), std::move(valueView)) {}

        virtual ~ReadIterableIndexedVectorDecorator() override {};

        /**
         * An iterator that provides read-only access to the indices stored in the vector.
         */
        typedef typename Vector::first_view_type::const_iterator index_const_iterator;

        /**
         * An iterator that provides read-only access to the values stored in the vector.
         */
        typedef typename Vector::second_view_type::const_iterator value_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the vector.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const {
            return Vector::firstView_.cbegin();
        }

        /**
         * Returns an `index_const_iterator` to the end of the vector.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const {
            return Vector::firstView_.cend();
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the vector.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const {
            return Vector::secondView_.cbegin();
        }

        /**
         * Returns a `value_const_iterator` to the end of the vector.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const {
            return Vector::secondView_.cend();
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
        WriteIterableIndexedVectorDecorator(typename Vector::first_view_type&& indexView,
                                            typename Vector::second_view_type&& valueView)
            : Vector(std::move(indexView), std::move(valueView)) {}

        virtual ~WriteIterableIndexedVectorDecorator() override {};

        /**
         * An iterator that provides access to the indices stored in the vector and allows to modify them.
         */
        typedef typename Vector::first_view_type::iterator index_iterator;

        /**
         * An iterator that provides access to the values stored in the vector and allows to modify them.
         */
        typedef typename Vector::second_view_type::iterator value_iterator;

        /**
         * Returns an `index_iterator` to the beginning of the vector.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin() {
            return Vector::firstView_.begin();
        }

        /**
         * Returns an `index_iterator` to the end of the vector.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end() {
            return Vector::firstView_.end();
        }

        /**
         * Returns a `value_iterator` to the beginning of the vector.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin() {
            return Vector::secondView_.begin();
        }

        /**
         * Returns a `value_iterator` to the end of the vector.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end() {
            return Vector::secondView_.end();
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
