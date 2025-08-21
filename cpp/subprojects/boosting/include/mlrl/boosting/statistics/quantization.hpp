/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/statistic.hpp"
#include "mlrl/boosting/data/view_statistic_non_decomposable_dense.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_sparse_set.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

#include <functional>
#include <memory>
#include <optional>

namespace boosting {

    // Forward declarations
    template<typename>
    class IQuantizationMatrix;

    /**
     * Defines an interface for all classes that implement a method for quantizing statistics about the quality of
     * predictions for training examples.
     */
    class IQuantization {
        public:

            virtual ~IQuantization() {}

            /**
             * A visitor function for handling quantization matrices that are backed by a view of the type
             * `CContiguousView`.
             */
            template<typename StatisticType>
            using DenseDecomposableMatrixVisitor =
              std::function<void(std::unique_ptr<IQuantizationMatrix<CContiguousView<Statistic<StatisticType>>>>&)>;

            /**
             * A visitor function for handling quantization matrices that are backed by a view of the type
             * `SparseSetView`.
             */
            template<typename StatisticType>
            using SparseDecomposableMatrixVisitor =
              std::function<void(std::unique_ptr<IQuantizationMatrix<SparseSetView<Statistic<StatisticType>>>>&)>;

            /**
             * A visitor function for handling quantization matrices that are backed by a view of the type
             * `DenseNonDecomposableStatisticView`.
             */
            template<typename StatisticType>
            using DenseNonDecomposableMatrixVisitor = std::function<void(
              std::unique_ptr<IQuantizationMatrix<DenseNonDecomposableStatisticView<StatisticType>>>&)>;

            /**
             * Invokes one of the given visitor functions, depending on which one is able to handle the type of the
             * matrix that is used for storing quantized scores.
             *
             * @param denseDecomposable32BitVisitor     An optional visitor function for handling quantization matrices
             *                                          that are backed by a view of the type
             *                                          `CContiguousView<Statistic<float32>>`
             * @param denseDecomposable64BitVisitor     An optional visitor function for handling quantization matrices
             *                                          that are backed by a view of the type
             *                                          `CContiguousView<Statistic<float64>>`
             * @param sparseDecomposable32BitVisitor    An optional visitor function for handling quantization matrices
             *                                          that are backed by a view of the type
             *                                          `SparseSetView<Statistic<float32>>`
             * @param sparseDecomposable64BitVisitor    An optional visitor function for handling quantization matrices
             *                                          that are backed by a view of the type
             *                                          `SparseSetView<Statistic<float64>>`
             * @param denseNonDecomposable32BitVisitor  An optional visitor function for handling quantization matrices
             *                                          that are backed by a view of the type
             *                                          `DenseNonDecomposableStatisticView<float32>`
             * @param denseNonDecomposable64BitVisitor  An optional visitor function for handling quantization matrices
             *                                          that are backed by a view of the type
             *                                          `DenseNonDecomposableStatisticView<float64>`
             */
            virtual void visitQuantizationMatrix(
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) = 0;
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
     * Defines an interfaces for all matrices for storing quantized statistics that are backed by a view.
     */
    template<typename View>
    class IQuantizationMatrix : public IQuantizationFactory {
        public:

            virtual ~IQuantizationMatrix() override {}

            /**
             * Quantifies all statistics that corresponds to the available outputs.
             *
             * @param outputIndicesBegin  A `CompleteIndexVector::const_iterator` to the start of the indices of the
             *                            outputs for which the statistics should be quantized
             * @param outputIndicesEnd    A `CompleteIndexVector::const_iterator` to the end of the indices of the
             *                            outputs for which the statistics should be quantized
             */
            virtual void quantize(CompleteIndexVector::const_iterator outputIndicesBegin,
                                  CompleteIndexVector::const_iterator outputIndicesEnd) = 0;

            /**
             * Quantifies all statistics that correspond to a certain subset of the outputs.
             *
             * @param outputIndicesBegin  A `PartialIndexVector::const_iterator` to the start of the indices of the
             *                            outputs for which the statistics should be quantized
             * @param outputIndicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices of the
             *                            outputs for which the statistics should be quantized
             */
            virtual void quantize(PartialIndexVector::const_iterator outputIndicesBegin,
                                  PartialIndexVector::const_iterator outputIndicesEnd) = 0;

            /**
             * The type of the view, the matrix is backed by.
             */
            typedef View view_type;

            /**
             * Returns the view, the matrix is backed by.
             *
             * @return A reference to an object of type `view_type`
             */
            virtual const view_type& getView() const = 0;
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
