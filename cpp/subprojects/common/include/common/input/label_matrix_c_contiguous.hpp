/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/input/label_matrix.hpp"


/**
 * Implements random read-only access to the labels of individual training examples that are stored in a pre-allocated
 * C-contiguous array.
 */
class CContiguousLabelMatrix final : public ILabelMatrix {

    private:

        CContiguousView<uint8> view_;

    public:

        /**
         * @param numRows   The number of rows in the label matrix
         * @param numCols   The number of columns in the label matrix
         * @param array     A pointer to a C-contiguous array of type `uint8` that stores the labels
         */
        CContiguousLabelMatrix(uint32 numRows, uint32 numCols, uint8* array);

        /**
         * An iterator that provides read-only access to the values in the label matrix.
         */
        typedef CContiguousView<uint8>::const_iterator value_const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning of the given row
         */
        value_const_iterator row_values_cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end of the given row
         */
        value_const_iterator row_values_cend(uint32 row) const;

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        std::unique_ptr<LabelVector> getLabelVector(uint32 row) const override;

        std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
            const IStatisticsProviderFactory& factory) const override;

        std::unique_ptr<IInstanceSubSampling> createInstanceSubSampling(
            const IInstanceSubSamplingFactory& factory) const override;

};
