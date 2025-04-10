/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/view_statistic_non_decomposable_dense.hpp"
#include "mlrl/boosting/losses/loss_decomposable.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all non-decomposable loss functions that can be used in classification problems.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class INonDecomposableClassificationLoss : virtual public IClassificationLoss<StatisticType>,
                                               virtual public IDecomposableClassificationLoss<StatisticType> {
        public:

            virtual ~INonDecomposableClassificationLoss() override {}

            /**
             * Updates the statistics of the example at a specific index.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `CContiguousView` that provides random access to
             *                      the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `CContiguousView` that stores the currently
             *                      predicted scores
             * @param statisticView A reference to an object of type `DenseNonDecomposableStatisticView` to be updated
             */
            virtual void updateNonDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<StatisticType>& scoreMatrix,
              DenseNonDecomposableStatisticView<StatisticType>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `BinaryCsrView` that provides row-wise access to
             *                      the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `CContiguousView` that stores the currently
             *                      predicted scores
             * @param statisticView A reference to an object of type `DenseNonDecomposableStatisticView` to be updated
             */
            virtual void updateNonDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<StatisticType>& scoreMatrix,
              DenseNonDecomposableStatisticView<StatisticType>& statisticView) const = 0;
    };

    /**
     * Defines an interface for all non-decomposable loss functions that can be used in regression problems.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class INonDecomposableRegressionLoss : virtual public IRegressionLoss<StatisticType>,
                                           virtual public IDecomposableRegressionLoss<StatisticType> {
        public:

            virtual ~INonDecomposableRegressionLoss() override {}

            /**
             * Updates the statistics of the example at a specific index.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param regressionMatrix  A reference to an object of type `CContiguousView` that provides random access
             *                          to the regression scores of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousView` that stores the currently
             *                          predicted scores
             * @param statisticView     A reference to an object of type `DenseNonDecomposableStatisticView` to be
             *                          updated
             */
            virtual void updateNonDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix,
              DenseNonDecomposableStatisticView<StatisticType>& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param regressionMatrix  A reference to an object of type `CsrView` that provides row-wise access to the
             *                          regression scores of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousView` that stores the currently
             *                          predicted scores
             * @param statisticView     A reference to an object of type `DenseNonDecomposableStatisticView` to be
             *                          updated
             */
            virtual void updateNonDecomposableStatistics(
              uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix,
              DenseNonDecomposableStatisticView<StatisticType>& statisticView) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type
     * `INonDecomposableClassificationLoss`.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class INonDecomposableClassificationLossFactory : public IDecomposableClassificationLossFactory<StatisticType> {
        public:

            virtual ~INonDecomposableClassificationLossFactory() override {}

            /**
             * Creates and returns a new object of type `INonDecomposableClassificationLoss`.
             *
             * @return An unique pointer to an object of type `INonDecomposableClassificationLoss` that has been created
             */
            virtual std::unique_ptr<INonDecomposableClassificationLoss<StatisticType>>
              createNonDecomposableClassificationLoss() const = 0;

            std::unique_ptr<IDecomposableClassificationLoss<StatisticType>> createDecomposableClassificationLoss()
              const override final {
                return this->createNonDecomposableClassificationLoss();
            }
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type
     * `INonDecomposableRegressionLoss`.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class INonDecomposableRegressionLossFactory : public IDecomposableRegressionLossFactory<StatisticType> {
        public:

            virtual ~INonDecomposableRegressionLossFactory() override {}

            /**
             * Creates and returns a new object of type `INonDecomposableRegressionLoss`.
             *
             * @return An unique pointer to an object of type `INonDecomposableRegressionLoss` that has been created
             */
            virtual std::unique_ptr<INonDecomposableRegressionLoss<StatisticType>> createNonDecomposableRegressionLoss()
              const = 0;

            std::unique_ptr<IDecomposableRegressionLoss<StatisticType>> createDecomposableRegressionLoss()
              const override final {
                return this->createNonDecomposableRegressionLoss();
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a non-decomposable loss function.
     */
    class INonDecomposableLossConfig : virtual public ILossConfig {
        public:

            virtual ~INonDecomposableLossConfig() override {}

            bool isDecomposable() const override final {
                return false;
            }

            bool isSparse() const override {
                return true;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a non-decomposable loss function that can be used in
     * classification problems.
     */
    class INonDecomposableClassificationLossConfig : virtual public IClassificationLossConfig,
                                                     virtual public INonDecomposableLossConfig {
        public:

            /**
             * Provides access to the interface of an `INonDecomposableClassificationLossConfig`, abstracting away
             * certain configuration options that have already been pre-determined.
             *
             * @tparam StatisticType The type that should be used for representing statistics
             */
            template<typename StatisticType>
            class IPreset : public IClassificationLossConfig::IPreset<StatisticType> {
                public:

                    virtual ~IPreset() override {}

                    /**
                     * Creates and returns a new object of type `INonDecomposableClassificationLossFactory` according to
                     * the specified configuration.
                     *
                     * @return An unique pointer to an object of type `INonDecomposableClassificationLossFactory` that
                     *         has been created
                     */
                    virtual std::unique_ptr<INonDecomposableClassificationLossFactory<StatisticType>>
                      createNonDecomposableClassificationLossFactory() const = 0;

                    std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>>
                      createClassificationEvaluationMeasureFactory() const override final {
                        return this->createNonDecomposableClassificationLossFactory();
                    }

                    std::unique_ptr<IDistanceMeasureFactory<StatisticType>> createDistanceMeasureFactory()
                      const override final {
                        return this->createNonDecomposableClassificationLossFactory();
                    }
            };

            virtual ~INonDecomposableClassificationLossConfig() override {}

            /**
             * Creates and returns a new object of type `IPreset<float32>`.
             *
             * @return An unique pointer to an object of type `IPreset<float32>` that has been created
             */
            virtual std::unique_ptr<IPreset<float32>> createNonDecomposable32BitClassificationPreset() const = 0;

            /**
             * Creates and returns a new object of type `IPreset<float64>`.
             *
             * @return An unique pointer to an object of type `IPreset<float64>` that has been created
             */
            virtual std::unique_ptr<IPreset<float64>> createNonDecomposable64BitClassificationPreset() const = 0;

            std::unique_ptr<IClassificationLossConfig::IPreset<float32>> create32BitClassificationPreset()
              const override final {
                return this->createNonDecomposable32BitClassificationPreset();
            }

            std::unique_ptr<IClassificationLossConfig::IPreset<float64>> create64BitClassificationPreset()
              const override final {
                return this->createNonDecomposable64BitClassificationPreset();
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a non-decomposable loss function that can be used in
     * regression problems.
     */
    class INonDecomposableRegressionLossConfig : virtual public IRegressionLossConfig,
                                                 virtual public INonDecomposableLossConfig {
        public:

            /**
             * Provides access to the interface of an `INonDecomposableRegressionLossConfig`, abstracting away certain
             * configuration options that have already been pre-determined.
             *
             * @tparam StatisticType The type that should be used for representing statistics
             */
            template<typename StatisticType>
            class IPreset : public IRegressionLossConfig::IPreset<StatisticType> {
                public:

                    virtual ~IPreset() override {}

                    /**
                     * Creates and returns a new object of type `INonDecomposableRegressionLossFactory` according to the
                     * specified configuration.
                     *
                     * @return An unique pointer to an object of type `INonDecomposableRegressionLossFactory` that has
                     *         been created
                     */
                    virtual std::unique_ptr<INonDecomposableRegressionLossFactory<StatisticType>>
                      createNonDecomposableRegressionLossFactory() const = 0;

                    std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>>
                      createRegressionEvaluationMeasureFactory() const override final {
                        return this->createNonDecomposableRegressionLossFactory();
                    }
            };

            virtual ~INonDecomposableRegressionLossConfig() override {}

            /**
             * Creates and returns a new object of type `IPreset<float32>`.
             *
             * @return An unique pointer to an object of type `IPreset<float32>` that has been created
             */
            virtual std::unique_ptr<IPreset<float32>> createNonDecomposable32BitRegressionPreset() const = 0;

            /**
             * Creates and returns a new object of type `IPreset<float64>`.
             *
             * @return An unique pointer to an object of type `IPreset<float64>` that has been created
             */
            virtual std::unique_ptr<IPreset<float64>> createNonDecomposable64BitRegressionPreset() const = 0;

            std::unique_ptr<IRegressionLossConfig::IPreset<float32>> create32BitRegressionPreset()
              const override final {
                return this->createNonDecomposable32BitRegressionPreset();
            }

            std::unique_ptr<IRegressionLossConfig::IPreset<float64>> create64BitRegressionPreset()
              const override final {
                return this->createNonDecomposable64BitRegressionPreset();
            }
    };

}
