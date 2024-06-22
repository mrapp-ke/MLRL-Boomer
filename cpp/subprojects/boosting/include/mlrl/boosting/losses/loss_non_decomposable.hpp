/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/view_statistic_non_decomposable_dense.hpp"
#include "mlrl/boosting/losses/loss_decomposable.hpp"

#include <memory>

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all (non-decomposable) loss functions that are applied example-wise.
     */
    class INonDecomposableLoss : public IDecomposableLoss {
        public:

            virtual ~INonDecomposableLoss() override {}

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
     * Defines an interface for all factories that allow to create instances of the type `INonDecomposableLoss`.
     */
    class INonDecomposableLossFactory : public IDecomposableLossFactory {
        public:

            virtual ~INonDecomposableLossFactory() override {}

            /**
             * Creates and returns a new object of type `INonDecomposableLoss`.
             *
             * @return An unique pointer to an object of type `INonDecomposableLoss` that has been created
             */
            virtual std::unique_ptr<INonDecomposableLoss> createNonDecomposableLoss() const = 0;

            std::unique_ptr<IDecomposableLoss> createDecomposableLoss() const override final {
                return this->createNonDecomposableLoss();
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a (non-decomposable) loss function that is applied
     * example-wise.
     */
    class INonDecomposableLossConfig : public ILossConfig {
        public:

            virtual ~INonDecomposableLossConfig() override {}

            /**
             * Creates and returns a new object of type `INonDecomposableLossFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `INonDecomposableLossFactory` that has been created
             */
            virtual std::unique_ptr<INonDecomposableLossFactory> createNonDecomposableLossFactory() const = 0;

            std::unique_ptr<IEvaluationMeasureFactory> createEvaluationMeasureFactory() const override final {
                return this->createNonDecomposableLossFactory();
            }

            std::unique_ptr<IDistanceMeasureFactory> createDistanceMeasureFactory() const override final {
                return this->createNonDecomposableLossFactory();
            }

            bool isDecomposable() const override final {
                return false;
            }

            bool isSparse() const override {
                return true;
            }
    };

}
