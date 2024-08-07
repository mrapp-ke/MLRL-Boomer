/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <iterator>

/**
 * An iterator adaptor that adapts an iterator of a binary sparse vector, which provides access to a fixed number of
 * indices in increasing order, such that it acts as a forward iterator that returns a boolean value for each possible
 * index, corresponding to the value of the element at the respecitive index.
 *
 * @tparam IndexIterator The type of the iterator that provides access to the indices of all dense elements explicitly
 *                       stored in the vector
 */
template<typename IndexIterator>
class BinarySparseForwardIterator final {
    private:

        IndexIterator indexIterator_;

        IndexIterator indicesEnd_;

        uint32 index_;

        uint32 iteratorIndex_;

    public:

        /**
         * @param indicesBegin  An iterator to the beginning of the indices of all dense elements explicitly stored in
         *                      the vector
         * @param indicesEnd    An iterator to the end of the indices of all dense elements explicitly stored in the
         *                      vector
         * @param startIndex    The index to start at
         */
        BinarySparseForwardIterator(IndexIterator indicesBegin, IndexIterator indicesEnd, uint32 startIndex)
            : indexIterator_(indicesBegin), indicesEnd_(indicesEnd), index_(startIndex),
              iteratorIndex_(indexIterator_ != indicesEnd_ ? *indexIterator_ : 0) {}

        /**
         * The type that is used to represent the difference between two iterators.
         */
        typedef int difference_type;

        /**
         * The type of the elements, the iterator provides access to.
         */
        typedef bool value_type;

        /**
         * The type of a pointer to an element, the iterator provides access to.
         */
        typedef const bool* pointer;

        /**
         * The type of a reference to an element, the iterator provides access to.
         */
        typedef bool& reference;

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
            return indexIterator_ != indicesEnd_ && iteratorIndex_ == index_;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        BinarySparseForwardIterator<IndexIterator>& operator++() {
            ++index_;

            if (indexIterator_ != indicesEnd_ && iteratorIndex_ < index_) {
                indexIterator_++;

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
        BinarySparseForwardIterator<IndexIterator>& operator++(int n) {
            index_++;

            if (indexIterator_ != indicesEnd_ && iteratorIndex_ < index_) {
                indexIterator_++;

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
        bool operator!=(const BinarySparseForwardIterator<IndexIterator>& rhs) const {
            return index_ != rhs.index_;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator==(const BinarySparseForwardIterator<IndexIterator>& rhs) const {
            return index_ == rhs.index_;
        }
};

/**
 * Creates and returns a new `BinarySparseForwardIterator`.
 *
 * @tparam IndexIterator    The type of the iterator that provides access to the indices of all dense elements
 *                          explicitly stored in the vector
 * @param indicesBegin      An iterator to the beginning of the indices of all dense elements explicitly stored in the
 *                          vector
 * @param indicesEnd        An iterator to the end of the indices of all dense elements explicitly stored in the vector
 * @param startIndex        The index to start at
 * @return                  A `BinarySparseForwardIterator` that has been created
 */
template<typename IndexIterator>
static inline BinarySparseForwardIterator<IndexIterator> createBinarySparseForwardIterator(IndexIterator indicesBegin,
                                                                                           IndexIterator indicesEnd,
                                                                                           uint32 startIndex = 0) {
    return BinarySparseForwardIterator<IndexIterator>(indicesBegin, indicesEnd, startIndex);
}
