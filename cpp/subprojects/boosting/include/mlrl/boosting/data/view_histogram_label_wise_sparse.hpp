/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/util/dll_exports.hpp"
#include "mlrl/common/data/triple.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_composite.hpp"
#include "mlrl/common/data/view_vector.hpp"

namespace boosting {

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * label-wise decomposable loss function and are stored in a pre-allocated histogram in the list of lists (LIL)
     * format.
     */
    class MLRLBOOSTING_API SparseLabelWiseHistogramView
        : public CompositeMatrix<AllocatedCContiguousView<Triple<float64>>, AllocatedVector<float64>> {
        public:

            /**
             * @param numRows   The number of rows in the view
             * @param numCols   The number of columns in the view
             */
            SparseLabelWiseHistogramView(uint32 numRows, uint32 numCols);

            /**
             * @param other A reference to an object of type `SparseLabelWiseHistogramView` that should be copied
             */
            SparseLabelWiseHistogramView(SparseLabelWiseHistogramView&& other);

            virtual ~SparseLabelWiseHistogramView() override {}

            /**
             * An iterator that provides read-only access to the gradients and Hessians.
             */
            typedef typename AllocatedCContiguousView<Triple<float64>>::value_const_iterator value_const_iterator;

            /**
             * An iterator that provides read-only access to the weights that correspond to individual bins.
             */
            typedef typename AllocatedVector<float64>::const_iterator weight_const_iterator;

            /**
             * Returns a `const_iterator` to the beginning of the gradients and Hessians at a specific row.
             *
             * @param row   The index of the row
             * @return      A `const_iterator` to the beginning of the row
             */
            value_const_iterator values_cbegin(uint32 row) const;

            /**
             * Returns a `const_iterator` to the end of the gradients and Hessians at a specific row.
             *
             * @param row   The index of the row
             * @return      A `const_iterator` to the end of the row
             */
            value_const_iterator values_cend(uint32 row) const;

            /**
             * Returns a `weight_const_iterator` to the beginning of the weights that correspond to individual bins.
             *
             * @return A `weight_const_iterator` to the beginning
             */
            weight_const_iterator weights_cbegin() const;

            /**
             * Returns a `weight_const_iterator` to the end of the weights that correspond to individual bins.
             *
             * @return A `weight_const_iterator` to the end
             */
            weight_const_iterator weights_cend() const;
    };

}