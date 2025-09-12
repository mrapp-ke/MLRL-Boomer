/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/view_statistic_decomposable_dense.hpp"
#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/common/data/view_matrix_csr.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all decomposable loss functions that can be used in classification problems.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class IDecomposableClassificationLoss : virtual public IClassificationLoss<StatisticType>,
                                            virtual public IClassificationEvaluationMeasure<StatisticType> {
        public:

            virtual ~IDecomposableClassificationLoss() override {}

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `CContiguousView` that provides random access to
             *                      the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `CContiguousView` that stores the currently
             *                      predicted scores
             * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView A reference to an object of type `DenseDecomposableStatisticView` to be updated
             */
            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, CompleteIndexVector::const_iterator indicesBegin,
              CompleteIndexVector::const_iterator indicesEnd,
              DenseDecomposableStatisticView<StatisticType>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `CContiguousView` that provides random access to
             *                      the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `CContiguousView` that stores the currently
             *                      predicted scores
             * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView A reference to an object of type `DenseDecomposableStatisticView` to be updated
             */
            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, PartialIndexVector::const_iterator indicesBegin,
              PartialIndexVector::const_iterator indicesEnd,
              DenseDecomposableStatisticView<StatisticType>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `BinaryCsrView` that provides row-wise access to
             *                      the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `CContiguousView` that stores the currently
             *                      predicted scores
             * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView A reference to an object of type `DenseDecomposableStatisticView` to be updated
             */
            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<StatisticType>& scoreMatrix,
              CompleteIndexVector::const_iterator indicesBegin, CompleteIndexVector::const_iterator indicesEnd,
              DenseDecomposableStatisticView<StatisticType>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of template type `BinaryCsrView` that provides row-wise
             *                      access to the ground truth of the training examples
             * @param scoreMatrix   A reference to an object of type `CContiguousView` that stores the currently
             *                      predicted scores
             * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView A reference to an object of type `DenseDecomposableStatisticView` to be updated
             */
            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<StatisticType>& scoreMatrix,
              PartialIndexVector::const_iterator indicesBegin, PartialIndexVector::const_iterator indicesEnd,
              DenseDecomposableStatisticView<StatisticType>& statisticView) const = 0;
    };

    /**
     * Defines an interface for all decomposable loss functions that can be used in regression problems.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class IDecomposableRegressionLoss : virtual public IRegressionLoss<StatisticType>,
                                        virtual public IRegressionEvaluationMeasure<StatisticType> {
        public:

            virtual ~IDecomposableRegressionLoss() override {}

            /**
             * Updates the statistics of the example at a specific index, considering only the outputs, whose indices
             * are provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param regressionMatrix  A reference to an object of template type `CContiguousView` that provides random
             *                          access to the regression scores of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousView` that stores the currently
             *                          predicted scores
             * @param indicesBegin      A `CompleteIndexVector::const_iterator` to the beginning of the output indices
             * @param indicesEnd        A `CompleteIndexVector::const_iterator` to the end of the output indices
             * @param statisticView     A reference to an object of type `DenseDecomposableStatisticView` to be updated
             */
            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, CompleteIndexVector::const_iterator indicesBegin,
              CompleteIndexVector::const_iterator indicesEnd,
              DenseDecomposableStatisticView<StatisticType>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the outputs, whose indices
             * are provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param regressionMatrix  A reference to an object of template type `CContiguousView` that provides random
             *                          access to the regression scores of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousView` that stores the currently
             *                          predicted scores
             * @param indicesBegin      A `PartialIndexVector::const_iterator` to the beginning of the output indices
             * @param indicesEnd        A `PartialIndexVector::const_iterator` to the end of the output indices
             * @param statisticView     A reference to an object of type `DenseDecomposableStatisticView` to be updated
             */
            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, PartialIndexVector::const_iterator indicesBegin,
              PartialIndexVector::const_iterator indicesEnd,
              DenseDecomposableStatisticView<StatisticType>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the outputs, whose indices
             * are provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param regressionMatrix  A reference to an object of type `CsrView` that provides row-wise access to the
             *                          regression scores of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousView` that stores the currently
             *                          predicted scores
             * @param indicesBegin      A `CompleteIndexVector::const_iterator` to the beginning of the output indices
             * @param indicesEnd        A `CompleteIndexVector::const_iterator` to the end of the output indices
             * @param statisticView     A reference to an object of type `DenseDecomposableStatisticView` to be updated
             */
            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, CompleteIndexVector::const_iterator indicesBegin,
              CompleteIndexVector::const_iterator indicesEnd,
              DenseDecomposableStatisticView<StatisticType>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the outputs, whose indices
             * are provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param regressionMatrix  A reference to an object of type `CsrView` that provides row-wise access to the
             *                          regression scores of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousView` that stores the currently
             *                          predicted scores
             * @param indicesBegin      A `PartialIndexVector::const_iterator` to the beginning of the output indices
             * @param indicesEnd        A `PartialIndexVector::const_iterator` to the end of the output indices
             * @param statisticView     A reference to an object of type `DenseDecomposableStatisticView` to be updated
             */
            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, PartialIndexVector::const_iterator indicesBegin,
              PartialIndexVector::const_iterator indicesEnd,
              DenseDecomposableStatisticView<StatisticType>& statisticView) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type
     * `IDecomposableClassificationLoss`.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class IDecomposableClassificationLossFactory
        : virtual public IClassificationEvaluationMeasureFactory<StatisticType>,
          virtual public IDistanceMeasureFactory<StatisticType> {
        public:

            virtual ~IDecomposableClassificationLossFactory() override {}

            /**
             * Creates and returns a new object of type `IDecomposableClassificationLoss`.
             *
             * @return An unique pointer to an object of type `IDecomposableClassificationLoss` that has been created
             */
            virtual std::unique_ptr<IDecomposableClassificationLoss<StatisticType>>
              createDecomposableClassificationLoss() const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type
     * `IDecomposableClassificationLoss`.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class IDecomposableRegressionLossFactory : public IRegressionEvaluationMeasureFactory<StatisticType> {
        public:

            virtual ~IDecomposableRegressionLossFactory() override {}

            /**
             * Creates and returns a new object of type `IDecomposableRegressionLoss`.
             *
             * @return An unique pointer to an object of type `IDecomposableRegressionLoss` that has been created
             */
            virtual std::unique_ptr<IDecomposableRegressionLoss<StatisticType>> createDecomposableRegressionLoss()
              const = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a decomposable loss function.
     */
    class IDecomposableLossConfig : virtual public ILossConfig {
        public:

            virtual ~IDecomposableLossConfig() override {}

            bool isDecomposable() const override final {
                return true;
            }

            bool isSparse() const override {
                return false;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a decomposable loss function that can be used in
     * classification problems.
     */
    class IDecomposableClassificationLossConfig : virtual public IDecomposableLossConfig,
                                                  virtual public IClassificationLossConfig {
        public:

            /**
             * Provides access to the interface of an `IDecomposableClassificationLossConfig`, abstracting away certain
             * configuration options that have already been pre-determined.
             *
             * @tparam StatisticType The type that should be used for representing statistics
             */
            template<typename StatisticType>
            class IPreset : public IClassificationLossConfig::IPreset<StatisticType> {
                public:

                    virtual ~IPreset() override {}

                    /**
                     * Creates and returns a new object of type `IDecomposableClassificationLossFactory` according to
                     * the specified configuration.
                     *
                     * @return An unique pointer to an object of type `IDecomposableClassificationLossFactory` that has
                     * been created
                     */
                    virtual std::unique_ptr<IDecomposableClassificationLossFactory<StatisticType>>
                      createDecomposableClassificationLossFactory() const = 0;

                    std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>>
                      createClassificationEvaluationMeasureFactory() const override final {
                        return this->createDecomposableClassificationLossFactory();
                    }

                    std::unique_ptr<IDistanceMeasureFactory<StatisticType>> createDistanceMeasureFactory()
                      const override final {
                        return this->createDecomposableClassificationLossFactory();
                    }
            };

            virtual ~IDecomposableClassificationLossConfig() override {}

            /**
             * Creates and returns a new object of type `IPreset<float32>`.
             *
             * @return An unique pointer to an object of type `IPreset<float32>` that has been created
             */
            virtual std::unique_ptr<IPreset<float32>> createDecomposable32BitClassificationPreset() const = 0;

            /**
             * Creates and returns a new object of type `IPreset<float64>`.
             *
             * @return An unique pointer to an object of type `IPreset<float64>` that has been created
             */
            virtual std::unique_ptr<IPreset<float64>> createDecomposable64BitClassificationPreset() const = 0;

            std::unique_ptr<IClassificationLossConfig::IPreset<float32>> create32BitClassificationPreset()
              const override final {
                return this->createDecomposable32BitClassificationPreset();
            }

            std::unique_ptr<IClassificationLossConfig::IPreset<float64>> create64BitClassificationPreset()
              const override final {
                return this->createDecomposable64BitClassificationPreset();
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a decomposable loss function that can be used in
     * regression problems.
     */
    class IDecomposableRegressionLossConfig : virtual public IDecomposableLossConfig,
                                              virtual public IRegressionLossConfig {
        public:

            /**
             * Provides access to the interface of an `IDecomposableRegressionLossConfig`, abstracting away certain
             * configuration options that have already been pre-determined.
             *
             * @tparam StatisticType The type that should be used for representing statistics
             */
            template<typename StatisticType>
            class IPreset : public IRegressionLossConfig::IPreset<StatisticType> {
                public:

                    virtual ~IPreset() override {}

                    /**
                     * Creates and returns a new object of type `IDecomposableRegressionLossFactory` according to the
                     * specified configuration.
                     *
                     * @return An unique pointer to an object of type `IDecomposableRegressionLossFactory` that has been
                     *         created
                     */
                    virtual std::unique_ptr<IDecomposableRegressionLossFactory<StatisticType>>
                      createDecomposableRegressionLossFactory() const = 0;

                    std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>>
                      createRegressionEvaluationMeasureFactory() const override final {
                        return this->createDecomposableRegressionLossFactory();
                    }
            };

            virtual ~IDecomposableRegressionLossConfig() override {}

            /**
             * Creates and returns a new object of type `IPreset<float32>`.
             *
             * @return An unique pointer to an object of type `IPreset<float32>` that has been created
             */
            virtual std::unique_ptr<IPreset<float32>> createDecomposable32BitRegressionPreset() const = 0;

            /**
             * Creates and returns a new object of type `IPreset<float64>`.
             *
             * @return An unique pointer to an object of type `IPreset<float64>` that has been created
             */
            virtual std::unique_ptr<IPreset<float64>> createDecomposable64BitRegressionPreset() const = 0;

            std::unique_ptr<IRegressionLossConfig::IPreset<float32>> create32BitRegressionPreset()
              const override final {
                return this->createDecomposable32BitRegressionPreset();
            }

            std::unique_ptr<IRegressionLossConfig::IPreset<float64>> create64BitRegressionPreset()
              const override final {
                return this->createDecomposable64BitRegressionPreset();
            }
    };

}
