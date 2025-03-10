/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/util/math.hpp"
#include "mlrl/common/data/view.hpp"

#include <iterator>

namespace boosting {

    /**
     * An iterator that provides access to the elements that correspond to the diagonal of a C-contiguous square matrix.
     *
     * @tparam T The type of the elements that are stored in the matrix
     */
    template<typename T>
    class DiagonalIterator final {
        private:

            View<T> view_;

            uint32 index_;

        public:

            /**
             * @param view  A `View` that provides access to the elements in the matrix
             * @param index The index on the diagonal to start at
             */
            DiagonalIterator(View<T> view, uint32 index) : view_(view), index_(index) {}

            /**
             * The type that is used to represent the difference between two iterators.
             */
            typedef int difference_type;

            /**
             * The type of the elements, the iterator provides access to.
             */
            typedef typename View<T>::value_type value_type;

            /**
             * The type of a pointer to an element, the iterator provides access to.
             */
            typedef typename View<T>::value_type* pointer;

            /**
             * The type of a reference to an element, the iterator provides access to.
             */
            typedef typename View<T>::value_type& reference;

            /**
             * The tag that specifies the capabilities of the iterator.
             */
            typedef std::random_access_iterator_tag iterator_category;

            /**
             * Returns the element at a specific index.
             *
             * @param index The index of the element to be returned
             * @return      The element at the given index
             */
            reference operator[](uint32 index) const {
                return view_.array[util::triangularNumber(index + 1) - 1];
            }

            /**
             * Returns the element at a specific index.
             *
             * @param index The index of the element to be returned
             * @return      The element at the given index
             */
            reference operator[](uint32 index) {
                return view_.array[util::triangularNumber(index + 1) - 1];
            }

            /**
             * Returns the element, the iterator currently refers to.
             *
             * @return The element, the iterator currently refers to
             */
            reference operator*() const {
                return view_.array[util::triangularNumber(index_ + 1) - 1];
            }

            /**
             * Returns the element, the iterator currently refers to.
             *
             * @return The element, the iterator currently refers to
             */
            reference operator*() {
                return view_.array[util::triangularNumber(index_ + 1) - 1];
            }

            /**
             * Returns an iterator to the next element.
             *
             * @return A reference to an iterator to the next element
             */
            DiagonalIterator<T>& operator++() {
                ++index_;
                return *this;
            }

            /**
             * Returns an iterator to the next element.
             *
             * @return A reference to an iterator to the next element
             */
            DiagonalIterator<T>& operator++(int n) {
                index_++;
                return *this;
            }

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator to the previous element
             */
            DiagonalIterator<T>& operator--() {
                --index_;
                return *this;
            }

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator to the previous element
             */
            DiagonalIterator<T>& operator--(int n) {
                index_--;
                return *this;
            }

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators do not refer to the same element, false otherwise
             */
            bool operator!=(const DiagonalIterator<T>& rhs) const {
                return index_ != rhs.index_;
            }

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators refer to the same element, false otherwise
             */
            bool operator==(const DiagonalIterator<T>& rhs) const {
                return index_ == rhs.index_;
            }

            /**
             * Returns the difference between this iterator and another one.
             *
             * @param rhs   A reference to another iterator
             * @return      The difference between the iterators
             */
            difference_type operator-(const DiagonalIterator<T>& rhs) const {
                return (difference_type) index_ - (difference_type) rhs.index_;
            }
    };

}
