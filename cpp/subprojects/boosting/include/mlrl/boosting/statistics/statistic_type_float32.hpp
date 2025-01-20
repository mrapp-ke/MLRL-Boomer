/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/statistic_type.hpp"

namespace boosting {

    /**
     * Allows to use 32-bit floating point values for representing statistics about the quality of predictions for
     * training examples.
     */
    class Float32StatisticsConfig final : public IStatisticTypeConfig {};

}
