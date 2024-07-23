/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <iterator>

/**
 * An iterator adaptor that adapts iterators of a sparse vector, which provide access to a fixed number of indices in
 * increasing order, as well as corresponding values, such that it acts as a forward iterator that returns a value for
 * each possible index, using sparse values for elements not explicitly stored in the vector.
 *
 * @tparam IndexIterator    The type of the iterator that provides access to the indices of all dense elements
 *                          explicitly stored in the vector
 * @tparam ValueIterator    The type of the iterator that provides access to the values of all dense elements explicitly
 *                          stored in the vector
 */
template<typename IndexIterator, typename ValueIterator>
class SparseForwardIterator final {
    private:

        IndexIterator indexIterator_;

        IndexIterator indicesEnd_;

        ValueIterator valueIterator_;

        ValueIterator valuesEnd_;

        uint32 index_;

        uint32 iteratorIndex_;

        typename std::iterator_traits<ValueIterator>::value_type sparseValue_;

    public:

        /**
         * @param indicesBegin  An iterator to the beginning of the indices of all dense elements explicitly stored in
         *                      the vector
         * @param indicesEnd    An iterator to the end of the indices of all dense elements explicitly stored in the
         *                      vector
         * @param valuesBegin   An iterator to the beginning of the values of all dense elements explicitly stored in
         *                      the vector
         * @param valuesEnd     An iterator to the end of the values of all dense elements explicitly stored in the
         *                      vector
         * @param startIndex    The index to start at
         * @param sparseValue   The value to be used for sparse elements not explicitly stored in the vector
         */
        SparseForwardIterator(IndexIterator indicesBegin, IndexIterator indicesEnd, ValueIterator valuesBegin,
                              ValueIterator valuesEnd, uint32 startIndex,
                              typename std::iterator_traits<ValueIterator>::value_type sparseValue)
            : indexIterator_(indicesBegin), indicesEnd_(indicesEnd), valueIterator_(valuesBegin), valuesEnd_(valuesEnd),
              index_(startIndex), iteratorIndex_(indexIterator_ != indicesEnd_ ? *indexIterator_ : 0),
              sparseValue_(sparseValue) {}

        /**
         * The type that is used to represent the difference between two iterators.
         */
        typedef int difference_type;

        /**
         * The type of the elements, the iterator provides access to.
         */
        typedef typename std::iterator_traits<ValueIterator>::value_type value_type;

        /**
         * The type of a pointer to an element, the iterator provides access to.
         */
        typedef const value_type* pointer;

        /**
         * The type of a reference to an element, the iterator provides access to.
         */
        typedef value_type& reference;

        /**
         * The tag that specifies the capabilities of the iterator.
         */
        typedef std::forward_iterator_tag iterator_category;

        /**
         * Returns the element, the iterator currently refers to.
         *
         * @return The element, the iterator currently refers to
         */
        value_type operator*() const {
            if (indexIterator_ != indicesEnd_ && iteratorIndex_ == index_) {
                return *valueIterator_;
            } else {
                return sparseValue_;
            }
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        SparseForwardIterator<IndexIterator, ValueIterator>& operator++() {
            ++index_;

            if (indexIterator_ != indicesEnd_ && iteratorIndex_ < index_) {
                indexIterator_++;
                valueIterator_++;

                if (indexIterator_ != indicesEnd_) {
                    iteratorIndex_ = *indexIterator_;
                }
            }

            return *this;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        SparseForwardIterator<IndexIterator, ValueIterator>& operator++(int n) {
            index_++;

            if (indexIterator_ != indicesEnd_ && iteratorIndex_ < index_) {
                indexIterator_++;
                valueIterator_++;

                if (indexIterator_ != indicesEnd_) {
                    iteratorIndex_ = *indexIterator_;
                }
            }

            return *this;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators do not refer to the same element, false otherwise
         */
        bool operator!=(const SparseForwardIterator<IndexIterator, ValueIterator>& rhs) const {
            return index_ != rhs.index_;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator==(const SparseForwardIterator<IndexIterator, ValueIterator>& rhs) const {
            return index_ == rhs.index_;
        }
};

/**
 * Creates and returns a new `SparseForwardIterator`.
 *
 * @tparam IndexIterator    The type of the iterator that provides access to the indices of all dense elements
 *                          explicitly stored in the vector
 * @tparam ValueIterator    The type of the iterator that provides access to the values of all dense elements explicitly
 *                          stored in the vector
 * @param indicesBegin      An iterator to the beginning of the indices of all dense elements explicitly stored in the
 *                          vector
 * @param indicesEnd        An iterator to the end of the indices of all dense elements explicitly stored in the vector
 * @param valuesBegin       An iterator to the beginning of the values of all dense elements explicitly stored in the
 *                          vector
 * @param valuesEnd         An iterator to the end of the values of all dense elements explicitly stored in the vector
 * @param startIndex        The index to start at
 * @param sparseValue       The value to be used for sparse elements not explicitly stored in the vector
 * @return                  A `SparseForwardIterator` that has been created
 */
template<typename IndexIterator, typename ValueIterator>
static inline SparseForwardIterator<IndexIterator, ValueIterator> createSparseForwardIterator(
  IndexIterator indicesBegin, IndexIterator indicesEnd, ValueIterator valuesBegin, ValueIterator valuesEnd,
  uint32 startIndex = 0, typename std::iterator_traits<ValueIterator>::value_type sparseValue = 0) {
    return SparseForwardIterator<IndexIterator, ValueIterator>(indicesBegin, indicesEnd, valuesBegin, valuesEnd,
                                                               startIndex, sparseValue);
}
