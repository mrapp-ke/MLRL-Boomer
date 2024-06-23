/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/label_matrix_row_wise.hpp"

#include <memory>

/**
 * Defines an interface for all label matrices that provide row-wise access to the labels of individual examples that
 * are stored in a C-contiguous array.
 */
class MLRLCOMMON_API ICContiguousLabelMatrix : public IRowWiseLabelMatrix {
    public:

        virtual ~ICContiguousLabelMatrix() override {}
};

/**
 * Creates and returns a new object of the type `ICContiguousLabelMatrix`.
 *
 * @param array     A pointer to a C-contiguous array of type `uint8` that stores the labels
 * @param numRows   The number of rows in the label matrix
 * @param numCols   The number of columns in the label matrix
 * @return          An unique pointer to an object of type `ICContiguousLabelMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICContiguousLabelMatrix> createCContiguousLabelMatrix(const uint8* array, uint32 numRows,
                                                                                     uint32 numCols);
