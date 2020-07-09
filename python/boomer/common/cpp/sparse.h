/**
 * Provides implementations of sparse matrices.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#include "arrays.h"


namespace sparse {

    /**
     * A sparse matrix that stores binary values using the dictionary of keys (DOK) format.
     */
    class BinaryDokMatrix {

    private:

        /**
         * The width of the matrix.
         */
        uint32 width_;

        /**
         * The height of the matrix.
         */
        uint32 height_;

    public:

        /**
         * Creates a new matrix width a specific width and height. Initially, all elements are set to zero.
         *
         * @param width     The width of the matrix
         * @param height    The height of the matrix
         */
        BinaryDokMatrix(uint32 width, uint32 height);

        /**
         * Returns the width of the matrix.
         *
         * @return The width of the matrix
         */
        uint32 getWidth();

        /**
         * Returns the height of the matrix.
         *
         * @return The height of the matrix
         */
        uint32 getHeight();

        /**
         * Returns the element at a specific position.
         *
         * @param rowIndex      The row of the element to be returned
         * @param columnIndex   The column of the element to be returned
         */
        uint8 getValue(uint32 row, uint32 column);

    };

}
