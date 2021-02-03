/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "loss_example_wise.h"


namespace boosting {

    /**
     * A multi-label variant of the logistic loss that is applied example-wise.
     */
    class ExampleWiseLogisticLoss final : public IExampleWiseLoss {

        public:

            void updateExampleWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                             const DenseNumericMatrix<float64>& scoreMatrix,
                                             DenseExampleWiseStatisticMatrix& statisticMatrix) const override;

            float64 evaluate(uint32 exampleIndex, const LabelVector& labelVector,
                             const CContiguousView<float64>& scoreMatrix) const override;

    };

}
