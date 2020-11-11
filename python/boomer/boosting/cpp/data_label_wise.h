/**
 * Provides classes that store the gradients and Hessians, that have been calculated using a label-wise decomposable
 * loss function, in matrices or vectors.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/types.h"
#include <cstdlib>


namespace boosting {

    /**
     * An one-dimensional vector that stores gradients and Hessians in C-contiguous arrays.
     */
    class DenseLabelWiseStatisticsVector {

        private:

            uint32 numElements_;

            float64* gradients_;

            float64* hessians_;

        public:

            /**
             * @param numElements The number of gradients and Hessians in the vector
             */
            DenseLabelWiseStatisticsVector(uint32 numElements)
                : DenseLabelWiseStatisticsVector(numElements, false) {

            }

            /**
             * @param numElements The number of gradients and Hessians in the vector
             * @param True, if all gradients and Hessians in the vector should be initialized with zero, false otherwise
             */
            DenseLabelWiseStatisticsVector(uint32 numElements, bool init)
                : numElements_(numElements),
                  gradients_((float64*) (init ? calloc(numElements, sizeof(float64))
                                              : malloc(numElements * sizeof(float64)))),
                  hessians_((float64*) (init ? calloc(numElements, sizeof(float64))
                                             : malloc(numElements * sizeof(float64)))) {

            }

            ~DenseLabelWiseStatisticsVector() {
                free(gradients_);
                free(hessians_);
            }

            typedef float64* gradient_iterator;

            typedef const float64* gradient_const_iterator;

            typedef float64* hessian_iterator;

            typedef const float64* hessian_const_iterator;

            /**
             * Returns a `gradient_iterator` to the beginning of the gradients.
             *
             * @return A `gradient_iterator` to the beginning
             */
            gradient_iterator gradients_begin() {
                return gradients_;
            }

            /**
             * Returns a `gradient_iterator` to the end of the gradients.
             *
             * @return A `gradient_iterator` to the end
             */
            gradient_iterator gradients_end() {
                return &gradients_[numElements_];
            }

            /**
             * Returns a `gradient_const_iterator` to the beginning of the gradients.
             *
             * @return A `gradient_const_iterator` to the beginning
             */
            gradient_const_iterator gradients_cbegin() const {
                return gradients_;
            }

            /**
             * Returns a `gradient_const_iterator` to the end of the gradients.
             *
             * @return A `gradient_const_iterator` to the end
             */
            gradient_const_iterator gradients_cend() const {
                return &gradients_[numElements_];
            }

            /**
             * Returns a `hessian_iterator` to the beginning of the Hessians.
             *
             * @return A `hessian_iterator` to the beginning
             */
            hessian_iterator hessians_begin() {
                return hessians_;
            }

            /**
             * Returns a `hessian_iterator` to the end of the Hessians.
             *
             * @return A `hessian_iterator` to the end
             */
            hessian_iterator hessians_end() {
                return &hessians_[numElements_];
            }

            /**
             * Returns a `hessian_const_iterator` to the beginning of the Hessians.
             *
             * @return A `hessian_const_iterator` to the beginning
             */
            hessian_const_iterator hessians_cbegin() const {
                return hessians_;
            }

            /**
             * Returns a `hessian_const_iterator` to the end of the Hessians.
             *
             * @return A `hessian_const_iterator` to the end
             */
            hessian_const_iterator hessians_cend() const {
                return &hessians_[numElements_];
            }

            /**
             * Returns the number of gradients and Hessians in the vector.
             *
             * @return The number of gradients and Hessians in the vector
             */
            uint32 getNumElements() const {
                return numElements_;
            }

            /**
             * Sets all gradients and Hessians in the vector to zero.
             */
            void setAllToZero() {
                for (uint32 i = 0; i < numElements_; i++) {
                    gradients_[i] = 0;
                    hessians_[i] = 0;
                }
            }

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians
             */
            void add(DenseLabelWiseStatisticsVector::gradient_const_iterator gradientsBegin,
                     DenseLabelWiseStatisticsVector::gradient_const_iterator gradientsEnd,
                     DenseLabelWiseStatisticsVector::hessian_const_iterator hessiansBegin,
                     DenseLabelWiseStatisticsVector::hessian_const_iterator hessiansEnd) {
                for (uint32 i = 0; i < numElements_; i++) {
                    gradients_[i] += gradientsBegin[i];
                    hessians_[i] += hessiansBegin[i];
                }
            }

            /**
             * Adds all gradients and Hessians in another vector to this vector. The gradients and Hessians to be added
             * are multiplied by a specific weight.
             *
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void add(DenseLabelWiseStatisticsVector::gradient_const_iterator gradientsBegin,
                     DenseLabelWiseStatisticsVector::gradient_const_iterator gradientsEnd,
                     DenseLabelWiseStatisticsVector::hessian_const_iterator hessiansBegin,
                     DenseLabelWiseStatisticsVector::hessian_const_iterator hessiansEnd, float64 weight) {
                for (uint32 i = 0; i < numElements_; i++) {
                    gradients_[i] += (gradientsBegin[i] * weight);
                    hessians_[i] += (hessiansBegin[i] * weight);
                }
            }

            /**
             * Subtracts all gradients and Hessians in another vector from this vector. The gradients and Hessians to be
             * subtracted are multiplied by a specific weight.
             *
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void subtract(DenseLabelWiseStatisticsVector::gradient_const_iterator gradientsBegin,
                          DenseLabelWiseStatisticsVector::gradient_const_iterator gradientsEnd,
                          DenseLabelWiseStatisticsVector::hessian_const_iterator hessiansBegin,
                          DenseLabelWiseStatisticsVector::hessian_const_iterator hessiansEnd, float64 weight) {
                for (uint32 i = 0; i < numElements_; i++) {
                    gradients_[i] -= (gradientsBegin[i] * weight);
                    hessians_[i] -= (hessiansBegin[i] * weight);
                }
            }

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a `FullIndexVector`,
             * to this vector. The gradients and Hessians to be added are multiplied by a specific weight.
             *
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians
             * @param indices           A reference to a `FullIndexVector' that provides access to the indices
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(DenseLabelWiseStatisticsVector::gradient_const_iterator gradientsBegin,
                             DenseLabelWiseStatisticsVector::gradient_const_iterator gradientsEnd,
                             DenseLabelWiseStatisticsVector::hessian_const_iterator hessiansBegin,
                             DenseLabelWiseStatisticsVector::hessian_const_iterator hessiansEnd,
                             const FullIndexVector& indices, float64 weight) {
                for (uint32 i = 0; i < numElements_; i++) {
                    gradients_[i] += (gradientsBegin[i] * weight);
                    hessians_[i] += (hessiansBegin[i] * weight);
                }
            }

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `PartialIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a specific
             * weight.
             *
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians
             * @param indices           A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(DenseLabelWiseStatisticsVector::gradient_const_iterator gradientsBegin,
                             DenseLabelWiseStatisticsVector::gradient_const_iterator gradientsEnd,
                             DenseLabelWiseStatisticsVector::hessian_const_iterator hessiansBegin,
                             DenseLabelWiseStatisticsVector::hessian_const_iterator hessiansEnd,
                             const PartialIndexVector& indices, float64 weight) {
                PartialIndexVector::const_iterator indexIterator = indices.cbegin();

                for (uint32 i = 0; i < numElements_; i++) {
                    uint32 index = indexIterator[i];
                    gradients_[i] += (gradientsBegin[index] * weight);
                    hessians_[i] += (hessiansBegin[index] * weight);
                }
            }

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors.
             *
             * @param firstGradientsBegin   A `gradient_const_iterator` to the beginning of the first gradients
             * @param firstGradientsEnd     A `gradient_const_iterator` to the end of the first gradients
             * @param firstHessiansBegin    A `hessian_const_iterator` to the beginning of the first Hessians
             * @param firstHessiansEnd      A `hessian_const_iterator` to the end of the first Hessians
             * @param secondGradientsBegin  A `gradient_const_iterator` to the beginning of the second gradients
             * @param secondGradientsEnd    A `gradient_const_iterator` to the end of the second gradients
             * @param secondHessiansBegin   A `hessian_const_iterator` to the beginning of the second Hessians
             * @param secondHessiansEnd     A `hessian_const_iterator` to the end of the second Hessians
             */
            void difference(DenseLabelWiseStatisticsVector::gradient_const_iterator firstGradientsBegin,
                            DenseLabelWiseStatisticsVector::gradient_const_iterator firstGradientsEnd,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator firstHessiansBegin,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator firstHessiansEnd,
                            DenseLabelWiseStatisticsVector::gradient_const_iterator secondGradientsBegin,
                            DenseLabelWiseStatisticsVector::gradient_const_iterator secondGradientsEnd,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator secondHessiansBegin,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator secondHessiansEnd) {
                for (uint32 i = 0; i < numElements_; i++) {
                    gradients_[i] = firstGradientsBegin[i] - secondGradientsBegin[i];
                    hessians_[i] = firstHessiansBegin[i] - secondHessiansBegin[i];
                }
            }

            /**
             * Sets the gradients and Hessians in this vector the difference `first - second` between the gradients and
             * Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `FullIndexVector`.
             *
             * @param firstGradientsBegin   A `gradient_const_iterator` to the beginning of the first gradients
             * @param firstGradientsEnd     A `gradient_const_iterator` to the end of the first gradients
             * @param firstHessiansBegin    A `hessian_const_iterator` to the beginning of the first Hessians
             * @param firstHessiansEnd      A `hessian_const_iterator` to the end of the first Hessians
             * @param firstIndices          A reference to an object of type `FullIndexVector` that provides access to
             *                              the indices
             * @param secondGradientsBegin  A `gradient_const_iterator` to the beginning of the second gradients
             * @param secondGradientsEnd    A `gradient_const_iterator` to the end of the second gradients
             * @param secondHessiansBegin   A `hessian_const_iterator` to the beginning of the second Hessians
             * @param secondHessiansEnd     A `hessian_const_iterator` to the end of the second Hessians
             */
            void difference(DenseLabelWiseStatisticsVector::gradient_const_iterator firstGradientsBegin,
                            DenseLabelWiseStatisticsVector::gradient_const_iterator firstGradientsEnd,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator firstHessiansBegin,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator firstHessiansEnd,
                            const FullIndexVector& firstIndices,
                            DenseLabelWiseStatisticsVector::gradient_const_iterator secondGradientsBegin,
                            DenseLabelWiseStatisticsVector::gradient_const_iterator secondGradientsEnd,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator secondHessiansBegin,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator secondHessiansEnd) {
                for (uint32 i = 0; i < numElements_; i++) {
                    gradients_[i] = firstGradientsBegin[i] - secondGradientsBegin[i];
                    hessians_[i] = firstHessiansBegin[i] - secondHessiansBegin[i];
                }
            }

            /**
             * Sets the gradients and Hessians in this vector the difference `first - second` between the gradients and
             * Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param firstGradientsBegin   A `gradient_const_iterator` to the beginning of the first gradients
             * @param firstGradientsEnd     A `gradient_const_iterator` to the end of the first gradients
             * @param firstHessiansBegin    A `hessian_const_iterator` to the beginning of the first Hessians
             * @param firstHessiansEnd      A `hessian_const_iterator` to the end of the first Hessians
             * @param firstIndices          A reference to an object of type `PartialIndexVector` that provides access
             *                              to the indices
             * @param secondGradientsBegin  A `gradient_const_iterator` to the beginning of the second gradients
             * @param secondGradientsEnd    A `gradient_const_iterator` to the end of the second gradients
             * @param secondHessiansBegin   A `hessian_const_iterator` to the beginning of the second Hessians
             * @param secondHessiansEnd     A `hessian_const_iterator` to the end of the second Hessians
             */
            void difference(DenseLabelWiseStatisticsVector::gradient_const_iterator firstGradientsBegin,
                            DenseLabelWiseStatisticsVector::gradient_const_iterator firstGradientsEnd,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator firstHessiansBegin,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator firstHessiansEnd,
                            const PartialIndexVector& firstIndices,
                            DenseLabelWiseStatisticsVector::gradient_const_iterator secondGradientsBegin,
                            DenseLabelWiseStatisticsVector::gradient_const_iterator secondGradientsEnd,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator secondHessiansBegin,
                            DenseLabelWiseStatisticsVector::hessian_const_iterator secondHessiansEnd) {
                PartialIndexVector::const_iterator firstIndexIterator = firstIndices.cbegin();

                for (uint32 i = 0; i < numElements_; i++) {
                    uint32 firstIndex = firstIndexIterator[i];
                    gradients_[i] = firstGradientsBegin[firstIndex] - secondGradientsBegin[i];
                    hessians_[i] = firstHessiansBegin[firstIndex] - secondHessiansBegin[i];
                }
            }

    };

}
