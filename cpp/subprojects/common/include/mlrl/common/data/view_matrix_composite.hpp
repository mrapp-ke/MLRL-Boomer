/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_composite.hpp"
#include "mlrl/common/data/view_matrix.hpp"

#include <utility>

/**
 * A matrix that is backed two two-dimensional views.
 *
 * @tparam FirstView    The type of the first view
 * @tparam SecondView   The type of the second view
 */
template<typename FirstView, typename SecondView>
class MLRLCOMMON_API CompositeMatrix : public Matrix,
                                       public CompositeView<FirstView, SecondView> {
    public:

        /**
         * @param firstView     The first view, the matrix should be backed by
         * @param secondView    The second view, the matrix should be backed by
         * @param numRows       The number of rows in the matrix
         * @param numCols       The number of columns in the matrix
         */
        CompositeMatrix(FirstView&& firstView, SecondView&& secondView, uint32 numRows, uint32 numCols)
            : Matrix(numRows, numCols),
              CompositeView<FirstView, SecondView>(std::move(firstView), std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `CompositeMatrix` that should be copied
         */
        CompositeMatrix(const CompositeMatrix<FirstView, SecondView>& other)
            : Matrix(other), CompositeView<FirstView, SecondView>(other) {}

        /**
         * @param other A reference to an object of type `CompositeMatrix` that should be moved
         */
        CompositeMatrix(CompositeMatrix<FirstView, SecondView>&& other)
            : Matrix(other), CompositeView<FirstView, SecondView>(std::move(other)) {}

        virtual ~CompositeMatrix() override {}

        /**
         * Sets all values stored in the view to zero.
         */
        void clear() {
            CompositeView<FirstView, SecondView>::firstView.clear();
            CompositeView<FirstView, SecondView>::secondView.clear();
        }
};
