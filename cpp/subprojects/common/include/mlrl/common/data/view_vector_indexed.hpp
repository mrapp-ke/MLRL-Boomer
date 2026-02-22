/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_composite.hpp"

#include <utility>

/**
 * A vector that is backed by two one-dimensional views of a specific size, storing indices and corresponding values.
 *
 * @tparam IndexView    The type of the view, the indices are backed by
 * @tparam ValueView    The type of the view, the values are backed by
 */
template<typename IndexView, typename ValueView>
class MLRLCOMMON_API IndexedVectorDecorator : public ViewDecorator<CompositeVector<IndexView, ValueView>> {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        explicit IndexedVectorDecorator(CompositeVector<IndexView, ValueView>&& view)
            : ViewDecorator<CompositeVector<IndexView, ValueView>>(std::move(view)) {}

        virtual ~IndexedVectorDecorator() override {}

        /**
         * The type of the indices that are stored in the vector.
         */
        using index_type = typename IndexView::value_type;

        /**
         * The type of the values that are stored in the vector.
         */
        using value_type = typename ValueView::value_type;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const {
            return this->view.firstView.numElements;
        }
};

/**
 * Provides access via iterators to indices and corresponding values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class MLRLCOMMON_API IterableIndexedVectorDecorator : public Vector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        explicit IterableIndexedVectorDecorator(typename Vector::view_type&& view) : Vector(std::move(view)) {}

        virtual ~IterableIndexedVectorDecorator() override {}

        /**
         * An iterator that provides read-only access to the indices stored in the vector.
         */
        using index_const_iterator = typename Vector::view_type::first_view_type::const_iterator;

        /**
         * An iterator that provides read-only access to the values stored in the vector.
         */
        using value_const_iterator = typename Vector::view_type::second_view_type::const_iterator;

        /**
         * An iterator that provides access to the indices stored in the vector and allows to modify them.
         */
        using index_iterator = typename Vector::view_type::first_view_type::iterator;

        /**
         * An iterator that provides access to the values stored in the vector and allows to modify them.
         */
        using value_iterator = typename Vector::view_type::second_view_type::iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the vector.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const {
            return Vector::view.firstView.cbegin();
        }

        /**
         * Returns an `index_const_iterator` to the end of the vector.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const {
            return Vector::view.firstView.cend();
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the vector.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const {
            return Vector::view.secondView.cbegin();
        }

        /**
         * Returns a `value_const_iterator` to the end of the vector.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const {
            return Vector::view.secondView.cend();
        }

        /**
         * Returns an `index_iterator` to the beginning of the vector.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin() {
            return Vector::view.firstView.begin();
        }

        /**
         * Returns an `index_iterator` to the end of the vector.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end() {
            return Vector::view.firstView.end();
        }

        /**
         * Returns a `value_iterator` to the beginning of the vector.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin() {
            return Vector::view.secondView.begin();
        }

        /**
         * Returns a `value_iterator` to the end of the vector.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end() {
            return Vector::view.secondView.end();
        }
};
