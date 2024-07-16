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
     */
    class INonDecomposableClassificationLoss : virtual public ILoss,
                                               virtual public IDecomposableClassificationLoss {
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
            virtual void updateNonDecomposableStatistics(uint32 exampleIndex,
                                                         const CContiguousView<const uint8>& labelMatrix,
                                                         const CContiguousView<float64>& scoreMatrix,
                                                         DenseNonDecomposableStatisticView& statisticView) const = 0;

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
            virtual void updateNonDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                         const CContiguousView<float64>& scoreMatrix,
                                                         DenseNonDecomposableStatisticView& statisticView) const = 0;
    };

    /**
     * Defines an interface for all non-decomposable loss functions that can be used in regression problems.
     */
    class INonDecomposableRegressionLoss : virtual public ILoss,
                                           virtual public IDecomposableRegressionLoss {
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
            virtual void updateNonDecomposableStatistics(uint32 exampleIndex,
                                                         const CContiguousView<const float32>& regressionMatrix,
                                                         const CContiguousView<float64>& scoreMatrix,
                                                         DenseNonDecomposableStatisticView& statisticView) const = 0;

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
            virtual void updateNonDecomposableStatistics(uint32 exampleIndex,
                                                         const CsrView<const float32>& regressionMatrix,
                                                         const CContiguousView<float64>& scoreMatrix,
                                                         DenseNonDecomposableStatisticView& statisticView) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type
     * `INonDecomposableClassificationLoss`.
     */
    class INonDecomposableClassificationLossFactory : public IDecomposableClassificationLossFactory {
        public:

            virtual ~INonDecomposableClassificationLossFactory() override {}

            /**
             * Creates and returns a new object of type `INonDecomposableClassificationLoss`.
             *
             * @return An unique pointer to an object of type `INonDecomposableClassificationLoss` that has been created
             */
            virtual std::unique_ptr<INonDecomposableClassificationLoss> createNonDecomposableClassificationLoss()
              const = 0;

            std::unique_ptr<IDecomposableClassificationLoss> createDecomposableClassificationLoss()
              const override final {
                return this->createNonDecomposableClassificationLoss();
            }
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type
     * `INonDecomposableRegressionLoss`.
     */
    class INonDecomposableRegressionLossFactory : public IDecomposableRegressionLossFactory {
        public:

            virtual ~INonDecomposableRegressionLossFactory() override {}

            /**
             * Creates and returns a new object of type `INonDecomposableRegressionLoss`.
             *
             * @return An unique pointer to an object of type `INonDecomposableRegressionLoss` that has been created
             */
            virtual std::unique_ptr<INonDecomposableRegressionLoss> createNonDecomposableRegressionLoss() const = 0;

            std::unique_ptr<IDecomposableRegressionLoss> createDecomposableRegressionLoss() const override final {
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

            virtual ~INonDecomposableClassificationLossConfig() override {}

            /**
             * Creates and returns a new object of type `INonDecomposableClassificationLossFactory` according to the
             * specified configuration.
             *
             * @return An unique pointer to an object of type `INonDecomposableClassificationLossFactory` that has been
             *         created
             */
            virtual std::unique_ptr<INonDecomposableClassificationLossFactory>
              createNonDecomposableClassificationLossFactory() const = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a non-decomposable loss function that can be used in
     * regression problems.
     */
    class INonDecomposableRegressionLossConfig : virtual public IRegressionLossConfig,
                                                 virtual public INonDecomposableLossConfig {
        public:

            virtual ~INonDecomposableRegressionLossConfig() override {}

            /**
             * Creates and returns a new object of type `INonDecomposableRegressionLossFactory` according to the
             * specified configuration.
             *
             * @return An unique pointer to an object of type `INonDecomposableRegressionLossFactory` that has been
             *         created
             */
            virtual std::unique_ptr<INonDecomposableRegressionLossFactory> createNonDecomposableRegressionLossFactory()
              const = 0;
    };

}
