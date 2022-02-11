/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/indexed_value.hpp"
#include <iterator>


/**
 * An iterator adaptor that adapts an iterator, which provides access to the elements of a sparse vector, consisting of
 * an index and a value that are stored by an `IndexedValue`, such that it acts as a forward iterator that returns only
 * those elements whose indices are provided by a second iterator.
 *
 * @tparam Iterator         The type of the iterator that provides access to the elements of a sparse vector
 * @tparam ValueType        The type of the values that are stored by the sparse vector
 * @tparam IndexIterator    The type of the iterator that provides access to the subset of indices
 */
template<typename Iterator, typename ValueType, typename IndexIterator>
class SparseSubsetForwardIterator {

    private:

        Iterator iterator_;

        Iterator end_;

        IndexIterator indexIterator_;

        IndexIterator indicesEnd_;

    public:

        /**
         * @param begin         An iterator to the beginning of the sparse vector
         * @param end           An iterator to the end of the sparse vector
         * @param indicesBegin  An iterator to the beginning of the subset of indices
         * @param indicesEnd    An iterator to the end of the subset of indices
         */
        SparseSubsetForwardIterator(Iterator begin, Iterator end, IndexIterator indicesBegin, IndexIterator indicesEnd)
            : iterator_(begin), end_(end), indexIterator_(indicesBegin), indicesEnd_(indicesEnd) {
            if (indexIterator_ != indicesEnd_) {
                if (iterator_ != end_) {
                    uint32 index = *indexIterator_;
                    uint32 currentIndex = (*iterator_).index;

                    CHECK: if (currentIndex < index) {
                        iterator_++;

                        if (iterator_ != end_) {
                            currentIndex = (*iterator_).index;
                            goto CHECK;
                        } else {
                            indexIterator_ = indicesEnd_;
                        }
                    } else if (currentIndex > index) {
                        indexIterator_++;

                        if (indexIterator_ != indicesEnd_) {
                            index = *indexIterator_;
                            goto CHECK;
                        }
                    }
                } else {
                    indexIterator_ = indicesEnd_;
                }
            }
        }

        /**
         * The type that is used to represent the difference between two iterators.
         */
        typedef int difference_type;

        /**
         * The type of the elements, the iterator provides access to.
         */
        typedef const IndexedValue<ValueType> value_type;

        /**
         * The type of a pointer to an element, the iterator provides access to.
         */
        typedef const IndexedValue<ValueType>* pointer;

        /**
         * The type of a reference to an element, the iterator provides access to.
         */
        typedef const IndexedValue<ValueType>& reference;

        /**
         * The tag that specifies the capabilities of the iterator.
         */
        typedef std::forward_iterator_tag iterator_category;

        /**
         * Returns the element, the iterator currently refers to.
         *
         * @return The element, the iterator currently refers to
         */
        reference operator*() const {
            return *iterator_;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        SparseSubsetForwardIterator<Iterator, ValueType, IndexIterator>& operator++() {
            indexIterator_++;

            if (indexIterator_ != indicesEnd_) {
                iterator_++;

                if (iterator_ != end_) {
                    uint32 index = *indexIterator_;
                    uint32 currentIndex = (*iterator_).index;

                    CHECK: if (currentIndex < index) {
                        iterator_++;

                        if (iterator_ != end_) {
                            currentIndex = (*iterator_).index;
                            goto CHECK;
                        } else {
                            indexIterator_ = indicesEnd_;
                        }
                    } else if (currentIndex > index) {
                        indexIterator_++;

                        if (indexIterator_ != indicesEnd_) {
                            index = *indexIterator_;
                            goto CHECK;
                        }
                    }
                } else {
                    indexIterator_ = indicesEnd_;
                }
            }

            return *this;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        SparseSubsetForwardIterator<Iterator, ValueType, IndexIterator>& operator++(int n) {
            indexIterator_++;

            if (indexIterator_ != indicesEnd_) {
                iterator_++;

                if (iterator_ != end_) {
                    uint32 index = *indexIterator_;
                    uint32 currentIndex = (*iterator_).index;

                    CHECK: if (currentIndex < index) {
                        iterator_++;

                        if (iterator_ != end_) {
                            currentIndex = (*iterator_).index;
                            goto CHECK;
                        } else {
                            indexIterator_ = indicesEnd_;
                        }
                    } else if (currentIndex > index) {
                        indexIterator_++;

                        if (indexIterator_ != indicesEnd_) {
                            index = *indexIterator_;
                            goto CHECK;
                        }
                    }
                } else {
                    indexIterator_ = indicesEnd_;
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
        bool operator!=(const SparseSubsetForwardIterator<Iterator, ValueType, IndexIterator>& rhs) const {
            return indexIterator_ != rhs.indexIterator_;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator==(const SparseSubsetForwardIterator<Iterator, ValueType, IndexIterator>& rhs) const {
            return indexIterator_ == rhs.indexIterator_;
        }

};

/**
 * Creates and returns a new `SparseSubsetForwardIterator`.
 *
 * @tparam Iterator         The type of the iterator that provides access to the elements of a sparse vector
 * @tparam ValueType        The type of the values that are stored by the sparse vector
 * @tparam IndexIterator    The type of the iterator that provides access to the subset of indices
 * @param begin             An iterator to the beginning of the sparse vector
 * @param end               An iterator to the end of the sparse vector
 * @param indicesBegin      An iterator to the beginning of the subset of indices
 * @param indicesEnd        An iterator to the end of the subset of indices
 * @return                  The `SparseSubsetForwardIterator` that has been created
 */
template<typename Iterator, typename ValueType, typename IndexIterator>
static inline SparseSubsetForwardIterator<Iterator, ValueType, IndexIterator> make_subset_forward_iterator(
        Iterator begin, Iterator end, IndexIterator indicesBegin, IndexIterator indicesEnd) {
    return SparseSubsetForwardIterator<Iterator, ValueType, IndexIterator>(begin, end, indicesBegin, indicesEnd);
}
