/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/view_statistic_non_decomposable_dense.hpp"
#include "mlrl/common/data/tuple.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_sparse_set.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

#include <functional>
#include <memory>

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
             * `CContiguousView<Tuple<float64>>`.
             */
            typedef std::function<void(std::unique_ptr<IQuantizationMatrix<CContiguousView<Tuple<float64>>>>&)>
              DenseDecomposableMatrixVisitor;

            /**
             * A visitor function for handling quantization matrices that are backed by a view of the type
             * `SparseSetView<Tuple<float64>>`.
             */
            typedef std::function<void(std::unique_ptr<IQuantizationMatrix<SparseSetView<Tuple<float64>>>>&)>
              SparseDecomposableMatrixVisitor;

            /**
             * A visitor function for handling quantization matrices that are backed by a view of the type
             * `DenseNonDecomposableStatisticView`.
             */
            typedef std::function<void(std::unique_ptr<IQuantizationMatrix<DenseNonDecomposableStatisticView>>&)>
              DenseNonDecomposableMatrixVisitor;

            /**
             * Invokes one of the given visitor functions, depending on which one is able to handle the type of the
             * matrix that is used for storing quantized scores.
             *
             * @param denseDecomposableMatrixVisitor    The visitor function for handling quantization matrices that are
             *                                          backed by a view of the type `CContiguousView<Tuple<float64>>`
             * @param sparseDecomposableMatrixVisitor   The visitor function for handling quantization matrices that are
             *                                          backed by a view of the type `SparseSetView<Tuple<float64>>`
             * @param denseNonDecomposableMatrixVisitor The visitor function for handling quantization matrices that are
             *                                          backed by a view of the type `DenseNonDecomposableStatisticView`
             */
            virtual void visitQuantizationMatrix(
              DenseDecomposableMatrixVisitor denseDecomposableMatrixVisitor,
              SparseDecomposableMatrixVisitor sparseDecomposableMatrixVisitor,
              DenseNonDecomposableMatrixVisitor denseNonDecomposableMatrixVisitor) = 0;

            /**
             * Quantifies all statistics that corresonds to the available outputs.
             *
             * @param outputIndices A reference to an object of type `ICompleteIndexVector` that stores the indices of
             *                      the output for which the statistics should be quantized
             */
            // TODO Remove
            virtual void quantize(const CompleteIndexVector& outputIndices) = 0;

            /**
             * Quantifies all statistics that correspond to a certain subset of the outputs.
             *
             * @param outputIndices A reference to an object of type `IPartialIndexVector` that stores the indices of
             *                      the output for which the statistics should be quantized
             */
            // TODO Remove
            virtual void quantize(const PartialIndexVector& outputIndices) = 0;
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
             * @param statisticMatrix   A reference to an object of type `CContiguousView` that stores the statistics to
             *                          be quantized
             * @return                  An unique pointer to an object of type `IQuantization` that has been created
             */
            virtual std::unique_ptr<IQuantization> create(
              const CContiguousView<Tuple<float64>>& statisticMatrix) const = 0;

            /**
             * Creates and returns a new object of type `IQuantization`.
             *
             * @param statisticMatrix   A reference to an object of type `SparseSetView` that stores the statistics to
             *                          be quantized
             * @return                  An unique pointer to an object of type `IQuantization` that has been created
             */
            virtual std::unique_ptr<IQuantization> create(
              const SparseSetView<Tuple<float64>>& statisticMatrix) const = 0;

            /**
             * Creates and returns a new object of type `IQuantization`.
             *
             * @param statisticMatrix   A reference to an object of type `DenseNonDecomposableStatisticView` that stores
             *                          the statistics to be quantized
             * @return                  An unique pointer to an object of type `IQuantization` that has been created
             */
            virtual std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView& statisticMatrix) const = 0;
    };

    /**
     * Defines an interfaces for all matrices for storing quantized statistics that are backed by a view.
     */
    template<typename View>
    class IQuantizationMatrix : public IQuantizationFactory {
        public:

            virtual ~IQuantizationMatrix() override {}

            /**
             * Quantifies all statistics that corresonds to the available outputs.
             *
             * @param outputIndices A reference to an object of type `ICompleteIndexVector` that stores the indices of
             *                      the output for which the statistics should be quantized
             */
            virtual void quantize(const CompleteIndexVector& outputIndices) = 0;

            /**
             * Quantifies all statistics that correspond to a certain subset of the outputs.
             *
             * @param outputIndices A reference to an object of type `IPartialIndexVector` that stores the indices of
             *                      the output for which the statistics should be quantized
             */
            virtual void quantize(const PartialIndexVector& outputIndices) = 0;

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
