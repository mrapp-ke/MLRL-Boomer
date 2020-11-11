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
                : numRows_(numRows), numCols_(numCols), numHessians_(triangularNumber(numCols))
                  gradients_((float64*) (init ? calloc(numRows * numCols, sizeof(float64))
                                              : malloc(numRows * numCols * sizeof(float64)))),
                  hessians_((float64*) (init ? calloc(numRows * numHessians_, sizeof(float64))
                                             : malloc(numRows * numHessians_ * sizeof(float64)))) {

            }
            
            ~DenseLabelWiseStatisticsMatrix() {
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

    };

}
