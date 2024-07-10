/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/output_matrix_row_wise.hpp"

#include <memory>

/**
 * Defines an interface for all regression matrices that provide access to the ground truth regression scores of
 * training examples.
 */
class MLRLCOMMON_API IRowWiseRegressionMatrix : public IRowWiseOutputMatrix {
    public:

        virtual ~IRowWiseRegressionMatrix() override {}
};
