/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_bit.hpp"
#include "mlrl/common/data/view_matrix.hpp"
#include "mlrl/common/util/bit_functions.hpp"

#include <utility>

/**
 * A two-dimensional view that provides random access to integer values, each with a specific number of bits, stored in
 * a dense matrix of a specific size.
 */
class BitMatrix : public BitView,
                  public Matrix {
    public:

        /**
         * @param array             A pointer to an array of type `uint32` that stores the values, the view should
         *                          provide access to
         * @param numRows           The number of rows in the view
         * @param numCols           The number of columns in the view
         * @param numBitsPerElement The number of bits per element in the view
         */
        BitMatrix(uint32* array, uint32 numRows, uint32 numCols, uint32 numBitsPerElement)
            : BitView(array, numRows * numCols, numBitsPerElement), Matrix(numRows, numCols) {}

        /**
         * @param other A const reference to an object of type `BitMatrix` that should be copied
         */
        BitMatrix(const BitMatrix& other) : BitView(other), Matrix(other) {}

        /**
         * @param other A reference to an object of type `BitMatrix` that should be moved
         */
        BitMatrix(BitMatrix&& other) : BitView(std::move(other)), Matrix(std::move(other)) {}

        virtual ~BitMatrix() override {}
};

/**
 * Allocates the memory, a `BitMatrix` provides access to.
 *
 * @tparam BitMatrix The type of the bit matrix
 */
template<typename BitMatrix>
class BitMatrixAllocator : public BitMatrix {
    public:

        /**
         * @param numRows           The number of rows in the bit matrix
         * @param numCols           The number of columns in the bit matrix
         * @param numBitsPerElement The number of bits per element in the bit matrix
         * @param init              True, if all elements in the bit matrix should be value-initialized, false otherwise
         */
        explicit BitMatrixAllocator(uint32 numRows, uint32 numCols, uint32 numBitsPerElement, bool init = false)
            : BitMatrix(
                util::allocateMemory<typename BitMatrix::value_type>(
                  util::getBitArraySize<typename BitMatrix::value_type>(numCols, numBitsPerElement) * numRows, init),
                numRows, numCols, numBitsPerElement) {}

        /**
         * @param other A reference to an object of type `BitMatrixAllocator` that should be copied
         */
        BitMatrixAllocator(const BitMatrixAllocator<BitMatrix>& other) : BitMatrix(other) {
            throw std::runtime_error("Objects of type BitMatrixAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `BitMatrixAllocator` that should be moved
         */
        BitMatrixAllocator(BitMatrixAllocator<BitMatrix>&& other) : BitMatrix(std::move(other)) {
            other.release();
        }

        virtual ~BitMatrixAllocator() override {
            util::freeMemory(BitMatrix::array);
        }
};

/**
 * Allocates the memory, a `BitMatrix` provides access to.
 */
typedef BitMatrixAllocator<BitMatrix> AllocatedBitMatrix;
