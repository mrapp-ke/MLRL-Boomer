/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

/**
 * A two-dimensional view that provides access to values stored in a matrix of a specific size.
 */
class MLRLCOMMON_API Matrix {
    public:

        /**
         * The number of rows in the view.
         */
        uint32 numRows;

        /**
         * The number of columns in the view.
         */
        uint32 numCols;

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        Matrix(uint32 numRows, uint32 numCols) : numRows(numRows), numCols(numCols) {}

        /**
         * @param other A const reference to an object of type `Matrix` that should be copied
         */
        Matrix(const Matrix& other) : numRows(other.numRows), numCols(other.numCols) {}

        /**
         * @param other A reference to an object of type `Matrix` that should be moved
         */
        Matrix(Matrix&& other) : numRows(other.numRows), numCols(other.numCols) {}

        virtual ~Matrix() {}
};

/**
 * A matrix that is backed by a two-dimensional view of a specific size.
 *
 * @tparam View The type of view, the matrix is backed by
 */
template<typename View>
class MLRLCOMMON_API MatrixDecorator : public ViewDecorator<View> {
    public:

        /**
         * @param view The view, the matrix should be backed by
         */
        MatrixDecorator(View&& view) : ViewDecorator<View>(std::move(view)) {}

        virtual ~MatrixDecorator() override {}

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows in the matrix
         */
        uint32 getNumRows() const {
            return ViewDecorator<View>::view.numRows;
        }

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns in the matrix
         */
        uint32 getNumCols() const {
            return ViewDecorator<View>::view.numCols;
        }
};
