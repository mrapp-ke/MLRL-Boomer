/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/output_matrix.hpp"

#include <memory>

// Forward declarations
class IStatisticsProvider;
class IStatisticsProviderFactory;

/**
 * Defines an interface for all output matrices that provide access to the ground truth of training examples.
 */
class MLRLCOMMON_API IRowWiseOutputMatrix : public IOutputMatrix {
    public:

        virtual ~IRowWiseOutputMatrix() override {}

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on the type of this output
         * matrix.
         *
         * @param factory   A reference to an object of type `IStatisticsProviderFactory` that should be used to create
         *                  the instance
         * @return          An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
          const IStatisticsProviderFactory& factory) const = 0;
};
