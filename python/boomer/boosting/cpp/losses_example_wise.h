/**
 * Implements different differentiable loss functions that are applied example-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/input_data.h"


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
             * @param gradients         A pointer to an array of type `float64`, shape `(num_labels)`, the gradients
             *                          that have been calculated should be written to. May contain arbitrary values
             * @param hessians          A pointer to an array of type `float64`, shape
             *                          `(num_labels * (num_labels + 1) / 2)` the Hessians that have been calculated
             *                          should be written to. May contain arbitrary values
             * @param predictedScore    A pointer to an array of type `float64`, shape `(num_labels)`, representing the
             *                          scores that are predicted for each label of the respective example
             */
            virtual void updateGradientsAndHessians(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                                    float64* gradients, float64* hessians,
                                                    const float64* predictedScores) const = 0;

    };

    /**
     * A multi-label variant of the logistic loss that is applied example-wise.
     */
    class ExampleWiseLogisticLossImpl : public IExampleWiseLoss {

        public:

            void updateGradientsAndHessians(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                            float64* gradients, float64* hessians,
                                            const float64* predictedScores) const override;

    };

}
