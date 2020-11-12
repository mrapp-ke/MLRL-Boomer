/**
 * Implements different differentiable loss functions that are applied example-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/input_data.h"
#include "data.h"
#include "data_example_wise.h"


namespace boosting {

    /**
     * A base class for all (non-decomposable) loss functions that are applied example-wise.
     */
    class IExampleWiseLoss {

        public:

            virtual ~IExampleWiseLoss() { };

            /**
             * Must be implemented by subclasses to calculate the gradients (first derivatives) and Hessians (second
             * derivatives) of the loss function for each label of a certain example.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be
             *                          calculated
             * @param labelMatrix       A reference to an object of type `IRandomAccessLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `DenseNumericMatrix` that stores the currently
             *                          predicted scores
             * @param statisticMatrix   A reference to an object of type `DenseExampleWiseStatisticsMatrix` to be
             *                          updated
             */
            virtual void updateStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                          const DenseNumericMatrix<float64>& scoreMatrix,
                                          DenseExampleWiseStatisticsMatrix& statisticMatrix) const = 0;

    };

    /**
     * A multi-label variant of the logistic loss that is applied example-wise.
     */
    class ExampleWiseLogisticLossImpl : public IExampleWiseLoss {

        public:

            void updateStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                  const DenseNumericMatrix<float64>& scoreMatrix,
                                  DenseExampleWiseStatisticsMatrix& statisticMatrix) const override;

    };

}
