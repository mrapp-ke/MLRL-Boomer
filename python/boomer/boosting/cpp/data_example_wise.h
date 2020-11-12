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
     * Calculates and returns the n-th triangular number, i.e., the number of elements in a n times n triangle.
     *
     * @param n A scalar of type `uint32`, representing the order of the triangular number
     * @return  A scalar of type `uint32`, representing the n-th triangular number
     */
    static inline uint32 triangularNumber(uint32 n) {
        return (n * (n + 1)) / 2;
    }

    /**
     * An one-dimensional vector that stores gradients and Hessians in C-contiguous arrays.
     */
    class DenseExampleWiseStatisticsVector {

        private:

            uint32 numElements_;

            uint32 numHessians_;

            float64* gradients_;

            float64* hessians_;

        public:

            /**
             * @param numElements The number of gradients and Hessians in the vector
             */
            DenseExampleWiseStatisticsVector(uint32 numElements)
                : DenseExampleWiseStatisticsVector(numElements, false) {

            }

            /**
             * @param numElements The number of gradients in the vector
             * @param True, if all gradients and Hessians in the vector should be initialized with zero, false otherwise
             */
            DenseExampleWiseStatisticsVector(uint32 numElements, bool init)
                : numElements_(numElements), numHessians_(triangularNumber(numElements)),
                  gradients_((float64*) (init ? calloc(numElements, sizeof(float64))
                                              : malloc(numElements * sizeof(float64)))),
                  hessians_((float64*) (init ? calloc(numHessians_, sizeof(float64))
                                             : malloc(numHessians_ * sizeof(float64)))) {

            }

            /**
             * @param vector A reference to an object of type `DenseExampleWiseStatisticsVector` to be copied
             */
            DenseExampleWiseStatisticsVector(const DenseExampleWiseStatisticsVector& vector)
                : DenseExampleWiseStatisticsVector(vector.numElements_) {
                for (uint32 i = 0; i < numElements_; i++) {
                    gradients_[i] = vector.gradients_[i];
                }

                for (uint32 i = 0; i < numHessians_; i++) {
                    hessians_[i] = vector.hessians_[i];
                }
            }

            ~DenseExampleWiseStatisticsVector() {
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
                return &hessians_[numHessians_];
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
                return &hessians_[numHessians_];
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
                }

                for (uint32 i = 0; i< numHessians_; i++) {
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
            void add(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                     hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd) {
                for (uint32 i = 0; i < numElements_; i++) {
                    gradients_[i] += gradientsBegin[i];
                }

                for (uint32 i = 0; i < numHessians_; i++) {
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
            void add(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                     hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd, float64 weight) {
                for (uint32 i = 0; i < numElements_; i++) {
                    gradients_[i] += (gradientsBegin[i] * weight);
                }

                for (uint32 i = 0; i < numHessians_; i++) {
                    hessians_[i] += (hessiansBegin[i] * weight);
                }
            }

    };

    /**
     * A two-dimensional matrix that stores gradients and Hessians in C-contiguous arrays.
     */
    class DenseExampleWiseStatisticsMatrix {

        private:

            uint32 numRows_;

            uint32 numCols_;

            uint32 numHessians_;

            float64* gradients_;

            float64* hessians_;

        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            DenseExampleWiseStatisticsMatrix(uint32 numRows, uint32 numCols)
                : DenseExampleWiseStatisticsMatrix(numRows, numCols, false) {

            }

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             * @param init      True, if all gradients and Hessians in the matrix should be initialized with zero, false
             *                  otherwise
             */
            DenseExampleWiseStatisticsMatrix(uint32 numRows, uint32 numCols, bool init)
                : numRows_(numRows), numCols_(numCols), numHessians_(triangularNumber(numCols)),
                  gradients_((float64*) (init ? calloc(numRows * numCols, sizeof(float64))
                                              : malloc(numRows * numCols * sizeof(float64)))),
                  hessians_((float64*) (init ? calloc(numRows * numHessians_, sizeof(float64))
                                             : malloc(numRows * numHessians_ * sizeof(float64)))) {

            }
            
            ~DenseExampleWiseStatisticsMatrix() {
                free(gradients_);
                free(hessians_);
            }
            
            typedef float64* gradient_iterator;
            
            typedef const float64* gradient_const_iterator;
            
            typedef float64* hessian_iterator;
            
            typedef const float64* hessian_const_iterator;
            
            /**
             * Returns a `gradient_iterator` to the beginning of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_iterator` to the beginning of the given row 
             */
            gradient_iterator gradients_row_begin(uint32 row) {
                return &gradients_[row * numCols_];
            }
            
            /**
             * Returns a `gradient_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_iterator` to the end of the given row 
             */
            gradient_iterator gradients_row_end(uint32 row) {
                return &gradients_[(row + 1) * numCols_];
            }
            
            /**
             * Returns a `gradient_const_iterator` to the beginning of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the beginning of the given row 
             */
            gradient_const_iterator gradients_row_cbegin(uint32 row) const {
                return &gradients_[row * numCols_];
            }
            
            /**
             * Returns a `gradient_const_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the end of the given row 
             */
            gradient_const_iterator gradients_row_cend(uint32 row) const {
                return &gradients_[(row + 1) * numCols_];   
            }

            /**
             * Returns a `hessian_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the beginning of the given row 
             */
            hessian_iterator hessians_row_begin(uint32 row) {
                return &hessians_[row * numHessians_];
            }
            
            /**
             * Returns a `hessian_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the end of the given row 
             */
            hessian_iterator hessians_row_end(uint32 row) {
                return &hessians_[(row + 1) * numHessians_];
            }
            
            /**
             * Returns a `hessian_const_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the beginning of the given row 
             */
            hessian_const_iterator hessians_row_cbegin(uint32 row) const {
                return &hessians_[row * numHessians_];
            }
            
            /**
             * Returns a `hessian_const_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the end of the given row 
             */
            hessian_const_iterator hessians_row_cend(uint32 row) const {
                return &hessians_[(row + 1) * numHessians_];
            }

            /**
             * Returns the number of rows in the matrix.
             *
             * @return The number of rows in the matrix
             */
            uint32 getNumRows() const {
                return numRows_;
            }

            /**
             * Returns the number of columns in the matrix.
             *
             * @return The number of columns in the matrix
             */
            uint32 getNumCols() const {
                return numCols_;
            }

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this matrix.
             *
             * @param row               The row
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients in the vector
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients in the vector
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians in the vector
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians in the vector
             */
            void addToRow(uint32 row, gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                          hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd) {
                uint32 offset = row * numCols_;

                for (uint32 i = 0; i < numCols_; i++) {
                    gradients_[offset + i] += gradientsBegin[i];
                }

                offset = row * numHessians_;

                for (uint32 i = 0; i < numHessians_; i++) {
                    hessians_[offset + i] += hessiansBegin[i];
                }
            }

    };

}
