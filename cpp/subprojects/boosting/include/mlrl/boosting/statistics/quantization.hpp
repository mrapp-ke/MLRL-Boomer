/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/statistic.hpp"
#include "mlrl/boosting/data/view_statistic_non_decomposable_dense.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_sparse_set.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that implement a method for quantizing statistics about the quality of
     * predictions for training examples.
     */
    class IQuantization {
        public:

            virtual ~IQuantization() {}
    };

    /**
     * Defines an interface for all factories that allows to create instances of the type `IQuantization`.
     */
    class IQuantizationFactory {
        public:

            virtual ~IQuantizationFactory() {}

            /**
             * Creates and returns a new object of type `IQuantization`.
             *
             * @param statisticMatrix   A reference to an object of type `CContiguousView<Statistic<float32>>` that
             *                          stores the statistics to be quantized
             * @return                  An unique pointer to an object of type `IQuantization` that has been created
             */
            virtual std::unique_ptr<IQuantization> create(
              const CContiguousView<Statistic<float32>>& statisticMatrix) const = 0;

            /**
             * Creates and returns a new object of type `IQuantization`.
             *
             * @param statisticMatrix   A reference to an object of type `CContiguousView<Statistic<float64>>` that
             *                          stores the statistics to be quantized
             * @return                  An unique pointer to an object of type `IQuantization` that has been created
             */
            virtual std::unique_ptr<IQuantization> create(
              const CContiguousView<Statistic<float64>>& statisticMatrix) const = 0;

            /**
             * Creates and returns a new object of type `IQuantization`.
             *
             * @param statisticMatrix   A reference to an object of type `SparseSetView<Statistic<float32>>` that stores
             *                          the statistics to be quantized
             * @return                  An unique pointer to an object of type `IQuantization` that has been created
             */
            virtual std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float32>>& statisticMatrix) const = 0;

            /**
             * Creates and returns a new object of type `IQuantization`.
             *
             * @param statisticMatrix   A reference to an object of type `SparseSetView<Statistic<float64>>` that stores
             *                          the statistics to be quantized
             * @return                  An unique pointer to an object of type `IQuantization` that has been created
             */
            virtual std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float64>>& statisticMatrix) const = 0;

            /**
             * Creates and returns a new object of type `IQuantization`.
             *
             * @param statisticMatrix   A reference to an object of type `DenseNonDecomposableStatisticView<float32>`
             *                          that stores the statistics to be quantized
             * @return                  An unique pointer to an object of type `IQuantization` that has been created
             */
            virtual std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float32>& statisticMatrix) const = 0;

            /**
             * Creates and returns a new object of type `IQuantization`.
             *
             * @param statisticMatrix   A reference to an object of type `DenseNonDecomposableStatisticView<float64>`
             *                          that stores the statistics to be quantized
             * @return                  An unique pointer to an object of type `IQuantization` that has been created
             */
            virtual std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float64>& statisticMatrix) const = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a method for quantizing statistics about the quality
     * of predictions for training examples.
     */
    class IQuantizationConfig {
        public:

            virtual ~IQuantizationConfig() {}

            /**
             * Creates and returns a new object of type `IQuantizationFactory` according to the specified configuration.
             *
             * @return An unique pointer to an object of type `IQuantizationFactory` that has been created
             */
            virtual std::unique_ptr<IQuantizationFactory> createQuantizationFactory() const = 0;
    };

}
