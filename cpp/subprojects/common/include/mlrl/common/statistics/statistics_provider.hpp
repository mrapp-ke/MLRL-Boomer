/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csr.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/common/statistics/statistics.hpp"

#include <memory>

/**
 * Provides access to an object of type `IStatistics`.
 */
class IStatisticsProvider {
    public:

        virtual ~IStatisticsProvider() {}

        /**
         * Returns an object of type `IStatistics`.
         *
         * @return A reference to an object of type `IStatistics`
         */
        virtual IStatistics& get() const = 0;

        /**
         * Switches the implementation that is used for calculating the predictions of rules, as well their overall
         * quality, to the one that should be used for learning regular rules.
         */
        virtual void switchToRegularRuleEvaluation() = 0;

        /**
         * Switches the implementation that is used for calculating the predictions of rules, as well as their overall
         * quality, to the one that should be used for pruning rules.
         */
        virtual void switchToPruningRuleEvaluation() = 0;
};

/**
 * Defines an interface for all classes that allow to create instances of the class `IStatisticsProvider`.
 */
class IStatisticsProviderFactory {
    public:

        virtual ~IStatisticsProviderFactory() {}

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on a label matrix that provides
         * random access to the labels of the training examples.
         *
         * @param labelMatrix   A reference to an object of type `CContiguousView` that provides random access to the
         *                      labels of the training examples
         * @return              An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> create(const CContiguousView<const uint8>& labelMatrix) const = 0;

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on a sparse label matrix that
         * provides row-wise access to the labels of the training examples.
         *
         * @param labelMatrix   A reference to an object of type `BinaryCsrView` that provides row-wise access to the
         *                      labels of the training examples
         * @return              An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> create(const BinaryCsrView& labelMatrix) const = 0;

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on a regression matrix that
         * provides random access to the regression scores of the training examples.
         *
         * @param regressionMatrix  A reference to an object of type `CContiguousView` that provides random access to
         *                          the regression scores of the training examples
         * @return                  An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> create(
          const CContiguousView<const float32>& regressionMatrix) const = 0;

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on a regression matrix that
         * provides random access to the regression scores of the training examples.
         *
         * @param regressionMatrix  A reference to an object of type `BinaryCsrView` that provides row-wise access to
         *                          the regression scores of the training examples
         * @return                  An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> create(const CsrView<const float32>& regressionMatrix) const = 0;
};
