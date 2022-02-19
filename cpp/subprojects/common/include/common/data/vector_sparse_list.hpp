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

        if (index > firstIndex) {
            vector.pop_front();
        } else if (index == firstIndex) {
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
static inline typename SparseListVector<T>::iterator insertNext(SparseListVector<T>& vector, uint32 index, T& value,
                                                                typename SparseListVector<T>::iterator begin) {
    typename SparseListVector<T>::iterator end = vector.end();
    typename SparseListVector<T>::iterator next = begin;
    next++;

    while (next != end) {
        IndexedValue<T>& nextEntry = *next;
        uint32 nextIndex = nextEntry.index;

        if (index > nextIndex) {
            next = vector.erase_after(begin);
        } else if (index == nextIndex) {
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
 * that one of them, referred to as `current`, points to the last element whose index is smaller or equal to a given
 * index and the other one, referred to as `previous`, points to the element's predecessor.
 *
 * @param previous  A reference to the first iterator to be advanced. It must point to the predecessor of `current`
 * @param current   A reference to the second iterator to be advanced. It must point to the successor of `previous`
 * @param end       An iterator to the end of the `SparseListVector`
 * @param index     The index until which the given iterators should be advanced
 * @return          The index of the element, the iterator `current` points to after advancing
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
                currentIndex = nextIndex;
                break;
            } else {
                break;
            }
        }
    }

    return currentIndex;
}

/**
 * Adds an element, consisting of an index and a value, to the corresponding element in a `SparseListVector`, whose
 * elements are sorted in increasing order by their indices. The search for the correct position starts at the element
 * that is pointed to by a given iterator.
 *
 * @tparam T        The type of the values that are stored by the `SparseListVector`
 * @param vector    A reference to an object of type `SparseListVector`, the element should be added to
 * @param previous  A reference to an iterator that points to the predecessor of `current`
 * @param current   A reference to an iterator that points to the element to start at
 * @param end       An iterator to the end of the `SparseListVector`
 * @param index     The index of the element to be added
 * @param value     The value of the element to be added
 */
template<typename T>
static inline void add(SparseListVector<T>& vector, typename SparseListVector<T>::iterator& previous,
                       typename SparseListVector<T>::iterator& current, typename SparseListVector<T>::iterator end,
                       uint32 index, const T& value) {
    uint32 currentIndex = advance<T>(previous, current, end, index);

    if (index == currentIndex) {
        (*current).value += value;
    } else if (index > currentIndex) {
        current = vector.emplace_after(current, index, value);
    } else {
        current = vector.emplace_after(previous, index, value);
    }

    previous = current;
    current++;
}

/**
 * Adds an element, consisting of an index and a value, to the corresponding element in a `SparseListVector`, whose
 * elements are sorted in increasing order by their indices. The search for the correct position starts at the beginning
 * of the vector.
 *
 * @tparam T        The type of the values that are stored by the `SparseListVector`
 * @param vector    A reference to an object of type `SparseListVector`, the element should be added to
 * @param begin     A reference to an iterator that points to the beginning of the `SparseListVector`
 * @param end       An iterator to the end of the `SparseListVector`
 * @param index     The index of the element to be added
 * @param value     The value of the element to be added
 * @return          An iterator that points to the successor of the modified element
 */
template<typename T>
static inline typename SparseListVector<T>::iterator addFirst(SparseListVector<T>& vector,
                                                              typename SparseListVector<T>::iterator& begin,
                                                              typename SparseListVector<T>::iterator end, uint32 index,
                                                              const T& value) {
    if (begin == end) {
        vector.emplace_front(index, value);
        begin = vector.begin();
    } else {
        IndexedValue<T>& firstEntry = *begin;
        uint32 firstIndex = firstEntry.index;

        if (index == firstIndex) {
            firstEntry.value += value;
        } else if (index < firstIndex) {
            vector.emplace_front(index, value);
            begin = vector.begin();
        } else {
            typename SparseListVector<T>::iterator current = begin;
            current++;

            if (current != end) {
                add<T>(vector, begin, current, end, index, value);
            } else {
                begin = vector.emplace_after(begin, index, value);
            }

            return current;
        }
    }

    typename SparseListVector<T>::iterator current = begin;
    current++;
    return current;
}
