/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/statistic.hpp"
#include "mlrl/boosting/losses/loss_decomposable.hpp"
#include "mlrl/common/data/view_matrix_sparse_set.hpp"
#include "mlrl/common/measures/measure_evaluation_sparse.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all decomposable loss functions that are suited for the use of sparse data structures.
     * To meet this requirement, the gradients and Hessians that are computed by the loss function should be zero, if
     * the prediction for a label is correct.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class ISparseDecomposableClassificationLoss : virtual public IDecomposableClassificationLoss<StatisticType>,
                                                  public ISparseEvaluationMeasure<StatisticType> {
        public:

            virtual ~ISparseDecomposableClassificationLoss() override {}

            // Keep functions from the parent class rather than hiding them
            using IDecomposableClassificationLoss<StatisticType>::updateDecomposableStatistics;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `CContiguousView` that provides random access to
             *                      the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `SparseSetView` that stores the currently predicted
             *                      scores
             * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView A reference to an object of type `SparseSetView` to be updated
             */
            virtual void updateDecomposableStatistics(uint32 exampleIndex,
                                                      const CContiguousView<const uint8>& labelMatrix,
                                                      const SparseSetView<StatisticType>& scoreMatrix,
                                                      CompleteIndexVector::const_iterator indicesBegin,
                                                      CompleteIndexVector::const_iterator indicesEnd,
                                                      SparseSetView<Statistic<StatisticType>>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `CContiguousView` that provides random access to
             *                      the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `SparseSetView` that stores the currently predicted
             *                      scores
             * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView A reference to an object of type `SparseSetView` to be updated
             */
            virtual void updateDecomposableStatistics(uint32 exampleIndex,
                                                      const CContiguousView<const uint8>& labelMatrix,
                                                      const SparseSetView<StatisticType>& scoreMatrix,
                                                      PartialIndexVector::const_iterator indicesBegin,
                                                      PartialIndexVector::const_iterator indicesEnd,
                                                      SparseSetView<Statistic<StatisticType>>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `BinaryCsrView` that provides row-wise access to
             *                      the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `SparseSetView` that stores the currently predicted
             *                      scores
             * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView A reference to an object of type `SparseSetView` to be updated
             */
            virtual void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                      const SparseSetView<StatisticType>& scoreMatrix,
                                                      CompleteIndexVector::const_iterator indicesBegin,
                                                      CompleteIndexVector::const_iterator indicesEnd,
                                                      SparseSetView<Statistic<StatisticType>>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `BinaryCsrView` that provides row-wise access to
             *                      the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `SparseSetView` that stores the currently predicted
             *                      scores
             * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView A reference to an object of type `SparseSetView` to be updated
             */
            virtual void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                      const SparseSetView<StatisticType>& scoreMatrix,
                                                      PartialIndexVector::const_iterator indicesBegin,
                                                      PartialIndexVector::const_iterator indicesEnd,
                                                      SparseSetView<Statistic<StatisticType>>& statisticView) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type
     * `ISparseDecomposableClassificationLoss`.
     */
    class ISparseDecomposableClassificationLossFactory : public IDecomposableClassificationLossFactory,
                                                         public ISparseEvaluationMeasureFactory<float64> {
        public:

            virtual ~ISparseDecomposableClassificationLossFactory() override {}

            /**
             * Creates and returns a new object of type `ISparseDecomposableClassificationLoss`.
             *
             * @return An unique pointer to an object of type `ISparseDecomposableClassificationLoss` that has been
             *         created
             */
            virtual std::unique_ptr<ISparseDecomposableClassificationLoss<float64>>
              createSparseDecomposableClassificationLoss() const = 0;

            /**
             * @see `IDecomposableClassificationLossFactory::createDecomposableClassificationLoss`
             */
            std::unique_ptr<IDecomposableClassificationLoss<float64>> createDecomposableClassificationLoss()
              const override final {
                return this->createSparseDecomposableClassificationLoss();
            }

            /**
             * @see `ISparseEvaluationMeasureFactory::createSparseEvaluationMeasure`
             */
            std::unique_ptr<ISparseEvaluationMeasure<float64>> createSparseEvaluationMeasure() const override final {
                return this->createSparseDecomposableClassificationLoss();
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a decomposable loss function that is suited for the
     * use of sparse data structures.
     */
    class ISparseDecomposableClassificationLossConfig : public IDecomposableClassificationLossConfig {
        public:

            virtual ~ISparseDecomposableClassificationLossConfig() override {}

            /**
             * Creates and returns a new object of type `ISparseDecomposableClassificationLossFactory` according to the
             * specified configuration.
             *
             * @return An unique pointer to an object of type `ISparseDecomposableClassificationLossFactory` that has
             *         been created
             */
            virtual std::unique_ptr<ISparseDecomposableClassificationLossFactory>
              createSparseDecomposableClassificationLossFactory() const = 0;

            /**
             * Creates and returns a new object of type `ISparseEvaluationMeasureFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `ISparseEvaluationMeasureFactory` that has been created
             */
            std::unique_ptr<ISparseEvaluationMeasureFactory<float64>> createSparseEvaluationMeasureFactory() const {
                return this->createSparseDecomposableClassificationLossFactory();
            }

            std::unique_ptr<IDecomposableClassificationLossFactory> createDecomposableClassificationLossFactory()
              const override final {
                return this->createSparseDecomposableClassificationLossFactory();
            }

            bool isSparse() const override final {
                return true;
            }
    };

}
