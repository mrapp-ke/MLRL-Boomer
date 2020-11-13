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
     * Defines an interface for all (non-decomposable) loss functions that are applied example-wise.
     */
    class IExampleWiseLoss {

        public:

            virtual ~IExampleWiseLoss() { };

            /**
             * Updates the statistics of the example at a specific index.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `IRandomAccessLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `DenseNumericMatrix` that stores the currently
             *                          predicted scores
             * @param statisticMatrix   A reference to an object of type `DenseExampleWiseStatisticMatrix` to be updated
             */
            virtual void updateExampleWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                                     const DenseNumericMatrix<float64>& scoreMatrix,
                                                     DenseExampleWiseStatisticMatrix& statisticMatrix) const = 0;

    };

    /**
     * A multi-label variant of the logistic loss that is applied example-wise.
     */
    class ExampleWiseLogisticLossImpl : public IExampleWiseLoss {

        public:

            void updateExampleWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                             const DenseNumericMatrix<float64>& scoreMatrix,
                                             DenseExampleWiseStatisticMatrix& statisticMatrix) const override;

    };

}
