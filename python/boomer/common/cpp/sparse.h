/**
 * Provides implementations of sparse matrices.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#include "arrays.h"
#include <unordered_set>
#include <utility>


namespace sparse {

    /**
     * Implements a hash function for pairs that store two integers of type `uint32`.
     */
    struct PairHash {

        inline std::size_t operator()(const std::pair<uint32, uint32> &v) const {
            return (((uint64) v.first) << 32) | ((uint64) v.second);
        }

    };

    /**
     * A sparse matrix that stores binary values using the dictionary of keys (DOK) format.
     */
    class BinaryDokMatrix {

        private:

            /**
             * An unordered set that stores pairs of rows and columns, indicating the positions of non-zero elements.
             */
            std::unordered_set<std::pair<uint32, uint32>, PairHash> data_;

        public:

            /**
             * Sets a non-zero value to element at a specific position.
             *
             * @param row       The row of the element to be set
             * @param column    The column of the element to be set
             */
            void addValue(uint32 row, uint32 column);

            /**
             * Returns the element at a specific position.
             *
             * @param row       The row of the element to be returned
             * @param column    The column of the element to be returned
             */
            uint8 getValue(uint32 row, uint32 column);

    };

}
