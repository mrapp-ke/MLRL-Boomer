/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/indexed_value.hpp"
#include <forward_list>


/**
 * An one-dimensional sparse vector that stores elements, consisting of an index and a value, in a linked list.
 *
 * @tparam T The type of the values that are stored in the vector
 */
template<typename T>
using SparseListVector = std::forward_list<IndexedValue<T>>;

// TODO Check if the following functions are used multiple times, or if they should be moved to the files that use them

/**
 * Inserts a new element, consisting of an index and value, into a `SparseListVector`, whose elements are sorted in
 * increasing order by their indices. The search for the correct position starts at the beginning of the vector. All
 * elements that are located between the start of the vector and the element to be inserted are removed from the vector.
 *
 * @tparam T        The type of the values that are stored by the `SparseListVector`
 * @param vector    A reference to an object of type `SparseListVector`, the new element should be inserted into
 * @param index     The index of the element to be inserted
 * @param value     The value of the element to be inserted
 * @return          An iterator to the element that has been inserted
 */
template<typename T>
static inline typename SparseListVector<T>::iterator insertNext(SparseListVector<T>& vector, uint32 index, T& value) {
    while (!vector.empty()) {
        typename SparseListVector<T>::iterator first = vector.begin();
        IndexedValue<T>& firstEntry = *first;
        uint32 firstIndex = firstEntry.index;

        if (firstIndex < index) {
            vector.pop_front();
        } else if (firstIndex == index) {
            firstEntry.value = value;
            return first;
        } else {
            break;
        }
    }

    vector.emplace_front(index, value);
    return vector.begin();
}

/**
 * Inserts a new element, consisting of an index and a value, into a `SparseListVector`, whose elements are sorted in
 * increasing order by their indices. The search for the correct position starts at the element after a given iterator.
 * All elements that are located between the given iterator and the element to be inserted are removed from the vector.
 *
 * @tparam T        The type of the values that are stored by the `SparseListVector`
 * @param vector    A reference to an object of type `SparseListVector`, the new element should be inserted into
 * @param index     The index of the element to be inserted
 * @param value     The value of the element to be inserted
 * @param begin     An iterator to the element after which the search for the correct position should start
 * @return          An iterator to the element that has been inserted
 */
template<typename T>
static inline typename SparseListVector<T>::iterator insertNext(SparseListVector<T>& vector, uint32 index,  T& value,
                                                                typename SparseListVector<T>::iterator begin) {
    typename SparseListVector<T>::iterator end = vector.end();
    typename SparseListVector<T>::iterator next = begin;
    next++;

    while (next != end) {
        IndexedValue<T>& nextEntry = *next;
        uint32 nextIndex = nextEntry.index;

        if (nextIndex < index) {
            next = vector.erase_after(begin);
        } else if (nextIndex == index) {
            nextEntry.value = value;
            return next;
        } else {
            break;
        }
    }

    return vector.emplace_after(begin, index, value);
}

/**
 * Advances two iterators of a `SparseListVector`, whose elements are sorted in increasing order by their indices, such
 * that one of them, referred to as `current`, points to the first element whose index is smaller or equal to a given
 * index and the other one, referred to as `previous`, points to the element's predecessor.
 *
 * @param previous  A reference to the first iterator to be advanced. It must point to the predecessor of `current`
 * @param current   A reference to the second iterator to be advanced. It must point to the successor of `previous`
 * @param end       An iterator to the end of the `SparseListVector`
 * @param index     The index until which the given iterators should be advanced
 * @return          The index of the element, the iterator "current" points to after advancing
 */
template<typename T>
static inline uint32 advance(typename SparseListVector<T>::iterator& previous,
                             typename SparseListVector<T>::iterator& current,
                             typename SparseListVector<T>::iterator end, uint32 index) {
    uint32 currentIndex = (*current).index;

    if (index > currentIndex) {
        typename SparseListVector<T>::iterator next = current;
        next++;

        while (next != end) {
            uint32 nextIndex = (*next).index;

            if (index > nextIndex) {
                previous = current;
                current = next;
                next++;
                currentIndex = nextIndex;
            } else if (index == nextIndex) {
                previous = current;
                current = next;
                next++;
                currentIndex = nextIndex;
                break;
            } else {
                break;
            }
        }
    }

    return currentIndex;
}
