/**
 * Provides utility functions that allow to apply commonly used operations on data that is stored in matrices or
 * vectors.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"
#include "indices.h"


namespace vector {

    /**
     * Adds all numeric data in a vector to another vector of equal length.
     *
     * @tparam T        The type of the numeric data
     * @param fromBegin A `DenseVector<T>::const_iterator` to the beginning of the values in the vector, the values
     *                  should be taken from
     * @param fromEnd   A `DenseVector<T>::const_iterator` to the end of the values in the vector, the values should be
     *                  taken from
     * @param to        A `DenseVector<T>::iterator` to the beginning of the values in the vector, the values should be
     *                  added to
     */
    template<class T>
    static inline void add(typename DenseVector<T>::const_iterator fromBegin,
                           typename DenseVector<T>::const_iterator fromEnd, typename DenseVector<T>::iterator to) {
        for (auto fromIterator = fromBegin; fromIterator != fromEnd; fromIterator++) {
            T value = *fromIterator;
            *to += value;
            to++;
        }
    }

    /**
     * Adds all numeric data in a vector to another vector of equal length. The values to be added are multiplied by a
     * specific weight.
     *
     * @tparam T        The type of the numeric data
     * @param fromBegin A `DenseVector<T>::const_iterator` to the beginning of the values in the vector, the values
     *                  should be taken from
     * @param fromEnd   A `DenseVector<T>::const_iterator` to the end of the values in the vector, the values should be
     *                  taken from
     * @param to        A `DenseVector<T>::iterator` to the beginning of the values in the vector, the values should be
     *                  added to
     * @param weight    The weight, the values to be added should be multiplied by
     */
    template<class T>
    static inline void add(typename DenseVector<T>::const_iterator fromBegin,
                           typename DenseVector<T>::const_iterator fromEnd, typename DenseVector<T>::iterator to,
                           T weight) {
        for (auto fromIterator = fromBegin; fromIterator != fromEnd; fromIterator++) {
            T value = *fromIterator;
            *to += (value * weight);
            to++;
        }
    }

    /**
     * Adds a subset of the numeric data in a vector, identified via indices, to another vector.
     *
     * @tparam T            The type of the numeric data
     * @param from          A `DenseVector<T>::const_iterator` to the beginning of the values in the vector, the values
     *                      should be taken from
     * @param to            A `DenseVector<T>::iterator` to the beginning of the values in the vector, the values should
     *                      be added to
     * @param indicesBegin  A `FullIndexVector::const_iterator` to the beginning of the indices
     * @param indicesEnd    A `FullIndexVector::const_iterator` to the end of the indices
     */
    template<class T>
    static inline void addFromSubset(typename DenseVector<T>::const_iterator from, typename DenseVector<T>::iterator to,
                                     FullIndexVector::const_iterator indicesBegin,
                                     FullIndexVector::const_iterator indicesEnd) {
        for (auto indexIterator = indicesBegin; indexIterator != indicesEnd; indexIterator++) {
            uint32 index = *indexIterator;
            T value = *from;
            to[index] += value;
            from++;
        }
    }

    /**
     * Adds a subset of the numeric data in a vector, identified via indices, to another vector.
     *
     * @tparam T            The type of the numeric data
     * @param from          A `DenseVector<T>::const_iterator` to the beginning of the values in the vector, the values
     *                      should be taken from
     * @param to            A `DenseVector<T>::iterator` to the beginning of the values in the vector, the values should
     *                      be added to
     * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the indices
     * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices
     */
    template<class T>
    static inline void addFromSubset(typename DenseVector<T>::const_iterator from, typename DenseVector<T>::iterator to,
                                     PartialIndexVector::const_iterator indicesBegin,
                                     PartialIndexVector::const_iterator indicesEnd) {
        for (auto indexIterator = indicesBegin; indexIterator != indicesEnd; indexIterator++) {
            uint32 index = *indexIterator;
            T value = *from;
            to[index] += value;
            from++;
        }
    }

    /**
     * Adds the numeric data in a vector to a subset of another vector, identified via indices. The values to be added
     * are multiplied by a specific weight.
     *
     * @tparam T            The type of the numeric data
     * @param from          A `DenseVector<T>::const_iterator` to the beginning of the values in the vector, the values
     *                      should be taken from
     * @param to            A `DenseVector<T>::iterator` to the beginning of the values in the vector, the values should
     *                      be added to
     * @param indicesBegin  A `FullIndexVector::const_iterator` to the beginning of the indices
     * @param indicesEnd    A `FullIndexVector::const_iterator` to the end of the indices
     * @param weight        The weight, the values to be added should be multiplied by
     */
    template<class T>
    static inline void addToSubset(typename DenseVector<T>::const_iterator from, typename DenseVector<T>::iterator to,
                                   FullIndexVector::const_iterator indicesBegin, FullIndexVector::const_iterator indicesEnd,
                                   T weight) {
        for (auto indexIterator = indicesBegin; indexIterator != indicesEnd; indexIterator++) {
            uint32 index = *indexIterator;
            T value = from[index];
            *to += (value * weight);
            to++;
        }
    }

    /**
     * Adds the numeric data in a vector to a subset of another vector, identified via indices. The values to be added
     * are multiplied by a specific weight.
     *
     * @tparam T            The type of the numeric data
     * @param from          A `DenseVector<T>::const_iterator` to the beginning of the values in the vector, the values
     *                      should be taken from
     * @param to            A `DenseVector<T>::iterator` to the beginning of the values in the vector, the values should
     *                      be added to
     * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the indices
     * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices
     * @param weight        The weight, the values to be added should be multiplied by
     */
    template<class T>
    static inline void addToSubset(typename DenseVector<T>::const_iterator from, typename DenseVector<T>::iterator to,
                                   PartialIndexVector::const_iterator indicesBegin,
                                   PartialIndexVector::const_iterator indicesEnd, T weight) {
        for (auto indexIterator = indicesBegin; indexIterator != indicesEnd; indexIterator++) {
            uint32 index = *indexIterator;
            T value = from[index];
            *to += (value * weight);
            to++;
        }
    }

}
