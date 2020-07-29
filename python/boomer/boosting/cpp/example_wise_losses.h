/**
 * Implements different differentiable loss functions that are applied example-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/statistics.h"


namespace losses {

    /**
     * A base class for all (non-decomposable) loss functions that are applied example-wise.
     */
    class AbstractExampleWiseLoss {

        public:

            /**
             * Frees the memory occupied by the loss function.
             */
            virtual ~AbstractExampleWiseLoss();

            /**
             * Must be implemented by subclasses to calculate the gradients (first derivatives) and Hessians (second
             * derivatives) of the loss function for each label of a certain example.
             *
             * @param labelMatrix       A pointer to an object of type `AbstractLabelMatrix` that provides random access
             *                          to the labels of the training examples
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be
             *                          calculated
             * @param predictedScore    A pointer to an array of type `float64`, shape `(num_labels)`, representing the
             *                          scores that are predicted for each label of the respective example
             * @param gradients         A pointer to an array of type `float64`, shape `(num_labels)`, the gradients
             *                          that have been calculated should be written to
             * @param hessians          A pointer to an array of type `float64`, shape
             *                          `(num_labels * (num_labels + 1) / 2)` the Hessians that have been calculated
             *                          should be written to
             */
            virtual void calculateGradientsAndHessians(statistics::AbstractLabelMatrix* labelMatrix, intp exampleIndex,
                                                       const float64* predictedScores, float64* gradients,
                                                       float64* hessians);

    };

    /**
     * A multi-label variant of the logistic loss that is applied example-wise.
     */
    class ExampleWiseLogisticLossImpl : public AbstractExampleWiseLoss {

        public:

            ~ExampleWiseLogisticLossImpl();

            void calculateGradientsAndHessians(statistics::AbstractLabelMatrix* labelMatrix, intp exampleIndex,
                                               const float64* predictedScores, float64* gradients,
                                               float64* hessians) override;

    };

}
