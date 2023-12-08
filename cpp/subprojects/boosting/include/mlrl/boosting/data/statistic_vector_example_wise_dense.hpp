/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/iterator/diagonal_iterator.hpp"
#include "mlrl/common/data/view_vector_composite.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

namespace boosting {

    /**
     * An one-dimensional vector that stores gradients and Hessians that have been calculated using a non-decomposable
     * loss function in C-contiguous arrays. For each element in the vector a single gradient, but multiple Hessians are
     * stored. In a vector that stores `n` gradients `(n * (n + 1)) / 2` Hessians are stored. The Hessians can be viewed
     * as a symmetric Hessian matrix with `n` rows and columns.
     */
    class DenseExampleWiseStatisticVector final
        : public ClearableCompositeVectorDecorator<
            CompositeViewDecorator<AllocatedVector<float64>, AllocatedVector<float64>>> {
        public:

            /**
             * @param numGradients The number of gradients in the vector
             * @param init         True, if all gradients and Hessians in the vector should be initialized with zero,
             *                     false otherwise
             */
            DenseExampleWiseStatisticVector(uint32 numGradients, bool init = false);

            /**
             * @param other A reference to an object of type `DenseExampleWiseStatisticVector` to be copied
             */
            DenseExampleWiseStatisticVector(const DenseExampleWiseStatisticVector& other);

            /**
             * An iterator that provides access to the gradients in the vector and allows to modify them.
             */
            typedef View<float64>::iterator gradient_iterator;

            /**
             * An iterator that provides read-only access to the gradients in the vector.
             */
            typedef View<float64>::const_iterator gradient_const_iterator;

            /**
             * An iterator that provides access to the Hessians in the vector and allows to modify them.
             */
            typedef View<float64>::iterator hessian_iterator;

            /**
             * An iterator that provides read-only access to the Hessians in the vector.
             */
            typedef View<float64>::const_iterator hessian_const_iterator;

            /**
             * An iterator that provides read-only access to the Hessians that correspond to the diagonal of the Hessian
             * matrix.
             */
            typedef DiagonalConstIterator<float64> hessian_diagonal_const_iterator;

            /**
             * Returns a `gradient_iterator` to the beginning of the gradients.
             *
             * @return A `gradient_iterator` to the beginning
             */
            gradient_iterator gradients_begin();

            /**
             * Returns a `gradient_iterator` to the end of the gradients.
             *
             * @return A `gradient_iterator` to the end
             */
            gradient_iterator gradients_end();

            /**
             * Returns a `gradient_const_iterator` to the beginning of the gradients.
             *
             * @return A `gradient_const_iterator` to the beginning
             */
            gradient_const_iterator gradients_cbegin() const;

            /**
             * Returns a `gradient_const_iterator` to the end of the gradients.
             *
             * @return A `gradient_const_iterator` to the end
             */
            gradient_const_iterator gradients_cend() const;

            /**
             * Returns a `hessian_iterator` to the beginning of the Hessians.
             *
             * @return A `hessian_iterator` to the beginning
             */
            hessian_iterator hessians_begin();

            /**
             * Returns a `hessian_iterator` to the end of the Hessians.
             *
             * @return A `hessian_iterator` to the end
             */
            hessian_iterator hessians_end();

            /**
             * Returns a `hessian_const_iterator` to the beginning of the Hessians.
             *
             * @return A `hessian_const_iterator` to the beginning
             */
            hessian_const_iterator hessians_cbegin() const;

            /**
             * Returns a `hessian_const_iterator` to the end of the Hessians.
             *
             * @return A `hessian_const_iterator` to the end
             */
            hessian_const_iterator hessians_cend() const;

            /**
             * Returns a `hessian_diagonal_const_iterator` to the beginning of the Hessians that correspond to the
             * diagonal of the Hessian matrix.
             *
             * @return A `hessian_diagonal_const_iterator` to the beginning
             */
            hessian_diagonal_const_iterator hessians_diagonal_cbegin() const;

            /**
             * Returns a `hessian_diagonal_const_iterator` to the end of the Hessians that correspond to the diagonal of
             * the Hessian matrix.
             *
             * @return A `hessian_diagonal_const_iterator` to the end
             */
            hessian_diagonal_const_iterator hessians_diagonal_cend() const;

            /**
             * Returns the number of gradients in the vector.
             *
             * @return The number of gradients
             */
            uint32 getNumGradients() const;

            /**
             * Returns the number of Hessians in the vector.
             *
             + @return The number of Hessians
             */
            uint32 getNumHessians() const;

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param gradientsBegin    An iterator to the beginning of the gradients
             * @param gradientsEnd      An iterator to the end of the gradients
             * @param hessiansBegin     An iterator to the beginning of the Hessians
             * @param hessiansEnd       An iterator to the end of the Hessians
             */
            void add(View<float64>::const_iterator gradientsBegin, View<float64>::const_iterator gradientsEnd,
                     View<float64>::const_iterator hessiansBegin, View<float64>::const_iterator hessiansEnd);

            /**
             * Adds all gradients and Hessians in another vector to this vector. The gradients and Hessians to be added
             * are multiplied by a specific weight.
             *
             * @param gradientsBegin    An iterator to the beginning of the gradients
             * @param gradientsEnd      An iterator to the end of the gradients
             * @param hessiansBegin     An iterator to the beginning of the Hessians
             * @param hessiansEnd       An iterator to the end of the Hessians
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void add(View<float64>::const_iterator gradientsBegin, View<float64>::const_iterator gradientsEnd,
                     View<float64>::const_iterator hessiansBegin, View<float64>::const_iterator hessiansEnd,
                     float64 weight);

            /**
             * Removes all gradients and Hessians in another vector from this vector.
             *
             * @param gradientsBegin    An iterator to the beginning of the gradients
             * @param gradientsEnd      An iterator to the end of the gradients
             * @param hessiansBegin     An iterator to the beginning of the Hessians
             * @param hessiansEnd       An iterator to the end of the Hessians
             */
            void remove(View<float64>::const_iterator gradientsBegin, View<float64>::const_iterator gradientsEnd,
                        View<float64>::const_iterator hessiansBegin, View<float64>::const_iterator hessiansEnd);

            /**
             * Removes all gradients and Hessians in another vector from this vector. The gradients and Hessians to be
             * removed are multiplied by a specific weight.
             *
             * @param gradientsBegin    An iterator to the beginning of the gradients
             * @param gradientsEnd      An iterator to the end of the gradients
             * @param hessiansBegin     An iterator to the beginning of the Hessians
             * @param hessiansEnd       An iterator to the end of the Hessians
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void remove(View<float64>::const_iterator gradientsBegin, View<float64>::const_iterator gradientsEnd,
                        View<float64>::const_iterator hessiansBegin, View<float64>::const_iterator hessiansEnd,
                        float64 weight);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector.
             *
             * @param gradientsBegin    An iterator to the beginning of the gradients
             * @param gradientsEnd      An iterator to the end of the gradients
             * @param hessiansBegin     An iterator to the beginning of the Hessians
             * @param hessiansEnd       An iterator to the end of the Hessians
             * @param indices           A reference to a `CompleteIndexVector` that provides access to the indices
             */
            void addToSubset(View<float64>::const_iterator gradientsBegin, View<float64>::const_iterator gradientsEnd,
                             View<float64>::const_iterator hessiansBegin, View<float64>::const_iterator hessiansEnd,
                             const CompleteIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `PartialIndexVector`, to this vector.
             *
             * @param gradientsBegin    An iterator to the beginning of the gradients
             * @param gradientsEnd      An iterator to the end of the gradients
             * @param hessiansBegin     An iterator to the beginning of the Hessians
             * @param hessiansEnd       An iterator to the end of the Hessians
             * @param indices           A reference to a `PartialIndexVector` that provides access to the indices
             */
            void addToSubset(View<float64>::const_iterator gradientsBegin, View<float64>::const_iterator gradientsEnd,
                             View<float64>::const_iterator hessiansBegin, View<float64>::const_iterator hessiansEnd,
                             const PartialIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a
             * specific weight.
             *
             * @param gradientsBegin    An iterator to the beginning of the gradients
             * @param gradientsEnd      An iterator to the end of the gradients
             * @param hessiansBegin     An iterator to the beginning of the Hessians
             * @param hessiansEnd       An iterator to the end of the Hessians
             * @param indices           A reference to a `CompleteIndexVector` that provides access to the indices
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(View<float64>::const_iterator gradientsBegin, View<float64>::const_iterator gradientsEnd,
                             View<float64>::const_iterator hessiansBegin, View<float64>::const_iterator hessiansEnd,
                             const CompleteIndexVector& indices, float64 weight);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `PartialIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a specific
             * weight.
             *
             * @param gradientsBegin    An iterator to the beginning of the gradients
             * @param gradientsEnd      An iterator to the end of the gradients
             * @param hessiansBegin     An iterator to the beginning of the Hessians
             * @param hessiansEnd       An iterator to the end of the Hessians
             * @param indices           A reference to a `PartialIndexVector` that provides access to the indices
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(View<float64>::const_iterator gradientsBegin, View<float64>::const_iterator gradientsEnd,
                             View<float64>::const_iterator hessiansBegin, View<float64>::const_iterator hessiansEnd,
                             const PartialIndexVector& indices, float64 weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `CompleteIndexVector`.
             *
             * @param firstGradientsBegin   An iterator to the beginning of the first gradients
             * @param firstGradientsEnd     An iterator to the end of the first gradients
             * @param firstHessiansBegin    An iterator to the beginning of the first Hessians
             * @param firstHessiansEnd      An iterator to the end of the first Hessians
             * @param firstIndices          A reference to an object of type `CompleteIndexVector` that provides access
             *                              to the indices
             * @param secondGradientsBegin  An iterator to the beginning of the second gradients
             * @param secondGradientsEnd    An iterator to the end of the second gradients
             * @param secondHessiansBegin   An iterator to the beginning of the second Hessians
             * @param secondHessiansEnd     An iterator to the end of the second Hessians
             */
            void difference(View<float64>::const_iterator firstGradientsBegin,
                            View<float64>::const_iterator firstGradientsEnd,
                            View<float64>::const_iterator firstHessiansBegin,
                            View<float64>::const_iterator firstHessiansEnd, const CompleteIndexVector& firstIndices,
                            View<float64>::const_iterator secondGradientsBegin,
                            View<float64>::const_iterator secondGradientsEnd,
                            View<float64>::const_iterator secondHessiansBegin,
                            View<float64>::const_iterator secondHessiansEnd);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param firstGradientsBegin   A iterator to the beginning of the first gradients
             * @param firstGradientsEnd     A iterator to the end of the first gradients
             * @param firstHessiansBegin    A iterator to the beginning of the first Hessians
             * @param firstHessiansEnd      A iterator to the end of the first Hessians
             * @param firstIndices          A reference to an object of type `PartialIndexVector` that provides access
             *                              to the indices
             * @param secondGradientsBegin  An iterator to the beginning of the second gradients
             * @param secondGradientsEnd    An iterator to the end of the second gradients
             * @param secondHessiansBegin   An iterator to the beginning of the second Hessians
             * @param secondHessiansEnd     An iterator to the end of the second Hessians
             */
            void difference(View<float64>::const_iterator firstGradientsBegin,
                            View<float64>::const_iterator firstGradientsEnd,
                            View<float64>::const_iterator firstHessiansBegin,
                            View<float64>::const_iterator firstHessiansEnd, const PartialIndexVector& firstIndices,
                            View<float64>::const_iterator secondGradientsBegin,
                            View<float64>::const_iterator secondGradientsEnd,
                            View<float64>::const_iterator secondHessiansBegin,
                            View<float64>::const_iterator secondHessiansEnd);
    };

}
