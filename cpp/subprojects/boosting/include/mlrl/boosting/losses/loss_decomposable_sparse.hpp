/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_decomposable.hpp"
#include "mlrl/common/data/tuple.hpp"
#include "mlrl/common/data/view_matrix_sparse_set.hpp"
#include "mlrl/common/measures/measure_evaluation_sparse.hpp"

namespace boosting {

    /**
     * Defines an interface for all decomposable loss functions that are suited for the use of sparse data structures.
     * To meet this requirement, the gradients and Hessians that are computed by the loss function should be zero, if
     * the prediction for a label is correct.
     */
    class ISparseDecomposableLoss : virtual public IDecomposableLoss,
                                    public ISparseEvaluationMeasure {
        public:

            virtual ~ISparseDecomposableLoss() override {}

            // Keep functions from the parent class rather than hiding them
            using IDecomposableLoss::updateDecomposableStatistics;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousView` that provides random access
             *                          to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `SparseSetView` that stores the currently
             *                          predicted scores
             * @param labelIndicesBegin A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseSetView` to be updated
             */
            virtual void updateDecomposableStatistics(uint32 exampleIndex,
                                                      const CContiguousView<const uint8>& labelMatrix,
                                                      const SparseSetView<float64>& scoreMatrix,
                                                      CompleteIndexVector::const_iterator labelIndicesBegin,
                                                      CompleteIndexVector::const_iterator labelIndicesEnd,
                                                      SparseSetView<Tuple<float64>>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousView` that provides random access
             *                          to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `SparseSetView` that stores the currently
             *                          predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseSetView` to be updated
             */
            virtual void updateDecomposableStatistics(uint32 exampleIndex,
                                                      const CContiguousView<const uint8>& labelMatrix,
                                                      const SparseSetView<float64>& scoreMatrix,
                                                      PartialIndexVector::const_iterator labelIndicesBegin,
                                                      PartialIndexVector::const_iterator labelIndicesEnd,
                                                      SparseSetView<Tuple<float64>>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `BinaryCsrView` that provides row-wise access
             *                          to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `SparseSetView` that stores the currently
             *                          predicted scores
             * @param labelIndicesBegin A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseSetView` to be updated
             */
            virtual void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                      const SparseSetView<float64>& scoreMatrix,
                                                      CompleteIndexVector::const_iterator labelIndicesBegin,
                                                      CompleteIndexVector::const_iterator labelIndicesEnd,
                                                      SparseSetView<Tuple<float64>>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `BinaryCsrView` that provides row-wise access
             *                          to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `SparseSetView` that stores the currently
             *                          predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseSetView` to be updated
             */
            virtual void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                      const SparseSetView<float64>& scoreMatrix,
                                                      PartialIndexVector::const_iterator labelIndicesBegin,
                                                      PartialIndexVector::const_iterator labelIndicesEnd,
                                                      SparseSetView<Tuple<float64>>& statisticView) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `ISparseDecomposableLoss`.
     */
    class ISparseDecomposableLossFactory : public IDecomposableLossFactory,
                                           public ISparseEvaluationMeasureFactory {
        public:

            virtual ~ISparseDecomposableLossFactory() override {}

            /**
             * Creates and returns a new object of type `ISparseDecomposableLoss`.
             *
             * @return An unique pointer to an object of type `ISparseDecomposableLoss` that has been created
             */
            virtual std::unique_ptr<ISparseDecomposableLoss> createSparseDecomposableLoss() const = 0;

            /**
             * @see `IDecomposableLossFactory::createDecomposableLoss`
             */
            std::unique_ptr<IDecomposableLoss> createDecomposableLoss() const override final {
                return this->createSparseDecomposableLoss();
            }

            /**
             * @see `ISparseEvaluationMeasureFactory::createSparseEvaluationMeasure`
             */
            std::unique_ptr<ISparseEvaluationMeasure> createSparseEvaluationMeasure() const override final {
                return this->createSparseDecomposableLoss();
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a decomposable loss function that is suited for the
     * use of sparse data structures.
     */
    class ISparseDecomposableLossConfig : public IDecomposableLossConfig {
        public:

            virtual ~ISparseDecomposableLossConfig() override {}

            /**
             * Creates and returns a new object of type `ISparseDecomposableLossFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `ISparseDecomposableLossFactory` that has been created
             */
            virtual std::unique_ptr<ISparseDecomposableLossFactory> createSparseDecomposableLossFactory() const = 0;

            /**
             * Creates and returns a new object of type `ISparseEvaluationMeasureFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `ISparseEvaluationMeasureFactory` that has been created
             */
            std::unique_ptr<ISparseEvaluationMeasureFactory> createSparseEvaluationMeasureFactory() const {
                return this->createSparseDecomposableLossFactory();
            }

            std::unique_ptr<IDecomposableLossFactory> createDecomposableLossFactory() const override final {
                return this->createSparseDecomposableLossFactory();
            }

            bool isSparse() const override final {
                return true;
            }
    };

}
