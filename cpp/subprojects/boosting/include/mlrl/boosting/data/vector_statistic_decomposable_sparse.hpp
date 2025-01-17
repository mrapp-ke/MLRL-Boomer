/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/statistic.hpp"
#include "mlrl/common/data/view_matrix_sparse_set.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

namespace boosting {

    /**
     * An individual label space statistic that consists of a gradient, a Hessian and a weight.
     *
     * @tparam StatisticType    The type of the gradient and Hessian
     * @tparam WeightType       The type of the weight
     */
    template<typename StatisticType, typename WeightType>
    struct SparseStatistic final {
        public:

            SparseStatistic() {}

            /**
             * @param gradient  The gradient
             * @param hessian   The Hessian
             * @param weight    The weight
             */
            SparseStatistic(StatisticType gradient, StatisticType hessian, WeightType weight)
                : gradient(gradient), hessian(hessian), weight(weight) {}

            /**
             * The gradient.
             */
            StatisticType gradient;

            /**
             * The Hessian.
             */
            StatisticType hessian;

            /**
             * The weight.
             */
            WeightType weight;

            /**
             * Adds the gradient, Hessian and weight of a given statistic to the gradient, Hessian and weight of this
             * statistic,
             *
             * @param rhs   A reference to the statistic, whose gradient, Hessian and weight should be added
             * @return      A reference to the modified statistic
             */
            SparseStatistic<StatisticType, WeightType>& operator+=(
              const SparseStatistic<StatisticType, WeightType>& rhs) {
                gradient += rhs.gradient;
                hessian += rhs.hessian;
                weight += rhs.weight;
                return *this;
            }

            /**
             * Creates and returns a new statistic that results from adding the gradient, Hessian and weight of a
             * specific statistic to the gradient, Hessian and weight of an existing statistic.
             *
             * @param lhs   The original statistic
             * @param rhs   A reference to the statistic, whose gradient, Hessian and weight should be added
             * @return      The statistic that has been created
             */
            friend SparseStatistic<StatisticType, WeightType> operator+(
              SparseStatistic<StatisticType, WeightType> lhs, const SparseStatistic<StatisticType, WeightType>& rhs) {
                lhs += rhs;
                return lhs;
            }

            /**
             * Subtracts the gradient, Hessian and weight of a given statistic from the gradient, Hessian and weight of
             * this statistic.
             *
             * @param rhs   A reference to the statistic, whose gradient, Hessian and weight should be subtracted
             * @return      A reference to the modified statistic
             */
            SparseStatistic<StatisticType, WeightType>& operator-=(
              const SparseStatistic<StatisticType, WeightType>& rhs) {
                gradient -= rhs.gradient;
                hessian -= rhs.hessian;
                weight -= rhs.weight;
                return *this;
            }

            /**
             * Creates and returns a new statistic that results from subtracting the gradient, Hessian and weight of a
             * specific statistic from the gradient, Hessian and weight of an existing statistic.
             *
             * @param lhs   The original statistic
             * @param rhs   A reference to the value to be subtracted
             * @return      The statistic that has been created
             */
            friend SparseStatistic<StatisticType, WeightType> operator-(
              SparseStatistic<StatisticType, WeightType> lhs, const SparseStatistic<StatisticType, WeightType>& rhs) {
                lhs -= rhs;
                return lhs;
            }
    };

    /**
     * An one-dimensional vector that stores aggregated gradients and Hessians that have been calculated using a
     * decomposable loss function in a C-contiguous array. For each element in the vector, a single gradient and
     * Hessian, as well as the sums of the weights of the aggregated gradients and Hessians, is stored.
     *
     * @tparam WeightType The type of the weights
     */
    template<typename WeightType>
    class SparseDecomposableStatisticVector final
        : public VectorDecorator<AllocatedVector<SparseStatistic<float64, WeightType>>> {
        private:

            /**
             * An iterator that provides random read-only access to the statistics in a
             * `SparseDecomposableStatisticVector`.
             */
            class ConstIterator final {
                private:

                    typename View<SparseStatistic<float64, WeightType>>::const_iterator iterator_;

                    const WeightType sumOfWeights_;

                public:

                    /**
                     * @param iterator      An iterator that provides access to the elements in a
                     *                      `SparseDecomposableStatisticVector`
                     * @param sumOfWeights  The sum of the weights of all statistics that have been added to the vector
                     */
                    ConstIterator(typename View<SparseStatistic<float64, WeightType>>::const_iterator iterator,
                                  WeightType sumOfWeights);

                    /**
                     * The type that is used to represent the difference between two iterators.
                     */
                    typedef int difference_type;

                    /**
                     * The type of the elements, the iterator provides access to.
                     */
                    typedef const Statistic<float64> value_type;

                    /**
                     * The type of a pointer to an element, the iterator provides access to.
                     */
                    typedef const Statistic<float64>* pointer;

                    /**
                     * The type of a reference to an element, the iterator provides access to.
                     */
                    typedef const Statistic<float64>& reference;

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
                    value_type operator[](uint32 index) const;

                    /**
                     * Returns the element, the iterator currently refers to.
                     *
                     * @return The element, the iterator currently refers to
                     */
                    value_type operator*() const;

                    /**
                     * Returns an iterator to the next element.
                     *
                     * @return A reference to an iterator that refers to the next element
                     */
                    ConstIterator& operator++();

                    /**
                     * Returns an iterator to the next element.
                     *
                     * @return A reference to an iterator that refers to the next element
                     */
                    ConstIterator& operator++(int n);

                    /**
                     * Returns an iterator to the previous element.
                     *
                     * @return A reference to an iterator that refers to the previous element
                     */
                    ConstIterator& operator--();

                    /**
                     * Returns an iterator to the previous element.
                     *
                     * @return A reference to an iterator that refers to the previous element
                     */
                    ConstIterator& operator--(int n);

                    /**
                     * Returns whether this iterator and another one refer to the same element.
                     *
                     * @param rhs   A reference to another iterator
                     * @return      True, if the iterators do not refer to the same element, false otherwise
                     */
                    bool operator!=(const ConstIterator& rhs) const;

                    /**
                     * Returns whether this iterator and another one refer to the same element.
                     *
                     * @param rhs   A reference to another iterator
                     * @return      True, if the iterators refer to the same element, false otherwise
                     */
                    bool operator==(const ConstIterator& rhs) const;

                    /**
                     * Returns the difference between this iterator and another one.
                     *
                     * @param rhs   A reference to another iterator
                     * @return      The difference between the iterators
                     */
                    difference_type operator-(const ConstIterator& rhs) const;
            };

            WeightType sumOfWeights_;

        public:

            /**
             * @param numElements   The number of gradients and Hessians in the vector
             * @param init          True, if all gradients and Hessians in the vector should be initialized with zero,
             *                      false otherwise
             */
            SparseDecomposableStatisticVector(uint32 numElements, bool init = false);

            /**
             * @param other A reference to an object of type `SparseDecomposableStatisticVector` to be copied
             */
            SparseDecomposableStatisticVector(const SparseDecomposableStatisticVector& other);

            /**
             * An iterator that provides read-only access to the elements in the vector.
             */
            typedef ConstIterator const_iterator;

            /**
             * Returns a `const_iterator` to the beginning of the vector.
             *
             * @return A `const_iterator` to the beginning
             */
            const_iterator cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the vector.
             *
             * @return A `const_iterator` to the end
             */
            const_iterator cend() const;

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param vector A reference to an object of type `SparseDecomposableStatisticVector` that stores the
             *               gradients and Hessians to be added to this vector
             */
            void add(const SparseDecomposableStatisticVector<WeightType>& vector);

            /**
             * Adds all gradients and Hessians in a single row of a `SparseSetView` to this vector.
             *
             * @param view  A reference to an object of type `SparseSetView` that stores the gradients and Hessians to
             *              be added to this vector
             * @param row   The index of the row to be added to this vector
             */
            void add(const SparseSetView<Statistic<float64>>& view, uint32 row);

            /**
             * Adds all gradients and Hessians in a single row of a `SparseSetView` to this vector. The gradients and
             * Hessians to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `SparseSetView` that stores the gradients and Hessians
             *                  to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void add(const SparseSetView<Statistic<float64>>& view, uint32 row, WeightType weight);

            /**
             * Removes all gradients and Hessians in a single row of a `SparseSetView` from this vector.
             *
             * @param view  A reference to an object of type `SparseSetView` that stores the gradients and Hessians to
             *              be removed from this vector
             * @param row   The index of the row to be removed from this vector
             */
            void remove(const SparseSetView<Statistic<float64>>& view, uint32 row);

            /**
             * Removes all gradients and Hessians in a single row of a `SparseSetView` from this vector. The gradients
             * and Hessians to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `SparseSetView` that stores the gradients and Hessians
             *                  to be removed from this vector
             * @param row       The index of the row to be removed from this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void remove(const SparseSetView<Statistic<float64>>& view, uint32 row, WeightType weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseSetView`, whose positions are given as a
             * `CompleteIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `SparseSetView` that stores the gradients and Hessians
             *                  to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             */
            void addToSubset(const SparseSetView<Statistic<float64>>& view, uint32 row,
                             const CompleteIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseSetView`, whose positions are given as a
             * `PartialIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `SparseSetView` that stores the gradients and Hessians
             *                  to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             */
            void addToSubset(const SparseSetView<Statistic<float64>>& view, uint32 row,
                             const PartialIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseSetView`, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a
             * specific weight.
             *
             * @param view      A reference to an object of type `SparseSetView` that stores the gradients and Hessians
             *                  to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const SparseSetView<Statistic<float64>>& view, uint32 row,
                             const CompleteIndexVector& indices, WeightType weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparsesetView`, whose positions are given as a
             * `PartialIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a specific
             * weight.
             *
             * @param view      A reference to an object of type `SparseSetView` that stores the gradients and Hessians
             *                  to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const SparseSetView<Statistic<float64>>& view, uint32 row,
                             const PartialIndexVector& indices, WeightType weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `CompleteIndexVector`.
             *
             * @param first         A reference to an object of type `SparseDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `SparseDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const SparseDecomposableStatisticVector<WeightType>& first,
                            const CompleteIndexVector& firstIndices,
                            const SparseDecomposableStatisticVector<WeightType>& second);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param first         A reference to an object of type `SparseDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `SparseDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const SparseDecomposableStatisticVector<WeightType>& first,
                            const PartialIndexVector& firstIndices,
                            const SparseDecomposableStatisticVector<WeightType>& second);

            /**
             * Sets all gradients and Hessians stored in this vector to zero.
             */
            void clear();
    };

}
