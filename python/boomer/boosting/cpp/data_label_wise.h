/**
 * Provides classes that store the gradients and Hessians, that have been calculated using a label-wise decomposable
 * loss function, in matrices or vectors.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/indices.h"


namespace boosting {

    /**
     * An one-dimensional vector that stores gradients and Hessians in C-contiguous arrays.
     */
    class DenseLabelWiseStatisticVector {

        private:

            uint32 numElements_;

            float64* gradients_;

            float64* hessians_;

        public:

            /**
             * @param numElements The number of gradients and Hessians in the vector
             */
            DenseLabelWiseStatisticVector(uint32 numElements);

            /**
             * @param numElements The number of gradients and Hessians in the vector
             * @param True, if all gradients and Hessians in the vector should be initialized with zero, false otherwise
             */
            DenseLabelWiseStatisticVector(uint32 numElements, bool init);

            /**
             * @param vector A reference to an object of type `DenseLabelWiseStatisticVector` to be copied
             */
            DenseLabelWiseStatisticVector(const DenseLabelWiseStatisticVector& vector);

            ~DenseLabelWiseStatisticVector();

            typedef float64* gradient_iterator;

            typedef const float64* gradient_const_iterator;

            typedef float64* hessian_iterator;

            typedef const float64* hessian_const_iterator;

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
             * Returns the number of gradients and Hessians in the vector.
             *
             * @return The number of gradients and Hessians in the vector
             */
            uint32 getNumElements() const;

            /**
             * Sets all gradients and Hessians in the vector to zero.
             */
            void setAllToZero();

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians
             */
            void add(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                     hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd);

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
                     hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd, float64 weight);

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
            void subtract(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                          hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd, float64 weight);

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
            void addToSubset(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                             hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd,
                             const FullIndexVector& indices, float64 weight);

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
            void addToSubset(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                             hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd,
                             const PartialIndexVector& indices, float64 weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
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
            void difference(gradient_const_iterator firstGradientsBegin, gradient_const_iterator firstGradientsEnd,
                            hessian_const_iterator firstHessiansBegin, hessian_const_iterator firstHessiansEnd,
                            const FullIndexVector& firstIndices, gradient_const_iterator secondGradientsBegin,
                            gradient_const_iterator secondGradientsEnd, hessian_const_iterator secondHessiansBegin,
                            hessian_const_iterator secondHessiansEnd);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
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
            void difference(gradient_const_iterator firstGradientsBegin, gradient_const_iterator firstGradientsEnd,
                            hessian_const_iterator firstHessiansBegin, hessian_const_iterator firstHessiansEnd,
                            const PartialIndexVector& firstIndices, gradient_const_iterator secondGradientsBegin,
                            gradient_const_iterator secondGradientsEnd, hessian_const_iterator secondHessiansBegin,
                            hessian_const_iterator secondHessiansEnd);

    };

    /**
     * A two-dimensional matrix that stores gradients and Hessians in C-contiguous arrays.
     */
    class DenseLabelWiseStatisticMatrix {

        private:

            uint32 numRows_;

            uint32 numCols_;

            float64* gradients_;

            float64* hessians_;

        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            DenseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols);

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             * @param init      True, if all gradients and Hessians in the matrix should be initialized with zero, false
             *                  otherwise
             */
            DenseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols, bool init);
            
            ~DenseLabelWiseStatisticMatrix();
            
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
            gradient_iterator gradients_row_begin(uint32 row);
            
            /**
             * Returns a `gradient_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_iterator` to the end of the given row 
             */
            gradient_iterator gradients_row_end(uint32 row);
            
            /**
             * Returns a `gradient_const_iterator` to the beginning of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the beginning of the given row 
             */
            gradient_const_iterator gradients_row_cbegin(uint32 row) const;
            
            /**
             * Returns a `gradient_const_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the end of the given row 
             */
            gradient_const_iterator gradients_row_cend(uint32 row) const;

            /**
             * Returns a `hessian_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the beginning of the given row 
             */
            hessian_iterator hessians_row_begin(uint32 row);
            
            /**
             * Returns a `hessian_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the end of the given row 
             */
            hessian_iterator hessians_row_end(uint32 row);
            
            /**
             * Returns a `hessian_const_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the beginning of the given row 
             */
            hessian_const_iterator hessians_row_cbegin(uint32 row) const;
            
            /**
             * Returns a `hessian_const_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the end of the given row 
             */
            hessian_const_iterator hessians_row_cend(uint32 row) const;

            /**
             * Returns the number of rows in the matrix.
             *
             * @return The number of rows in the matrix
             */
            uint32 getNumRows() const;

            /**
             * Returns the number of columns in the matrix.
             *
             * @return The number of columns in the matrix
             */
            uint32 getNumCols() const;

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
                          hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd);

    };

}
