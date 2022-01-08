/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/data/view_csr_binary.hpp"
#include "common/statistics/statistics_provider.hpp"


/**
 * Defines an interface for all classes that allow to create instances of the class `IStatisticsProvider`.
 */
class IStatisticsProviderFactory {

    public:

        virtual ~IStatisticsProviderFactory() { };

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on a label matrix that provides
         * random access to the labels of the training examples.
         *
         * @param labelMatrix   A reference to an object of type `CContiguousConstView` that provides random access to
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> create(
            const CContiguousConstView<const uint8>& labelMatrix) const = 0;

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on a sparse label matrix that
         * provides row-wise access to the labels of the training examples.
         *
         * @param labelMatrix   A reference to an object of type `BinaryCsrConstView` that provides row-wise access to
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> create(const BinaryCsrConstView& labelMatrix) const = 0;

};
