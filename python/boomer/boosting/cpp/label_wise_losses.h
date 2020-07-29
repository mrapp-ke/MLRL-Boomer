/**
 * Implements different differentiable loss functions that are applied label-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/statistics.h"
#include <utility>


namespace losses {

    /**
     * An abstract base class for all (decomposable) loss functions that are applied label-wise.
     */
    class AbstractLabelWiseLoss {

        public:

            /**
             * Frees the memory occupied by the loss function.
             */
            ~AbstractLabelWiseLoss();

        /**
         * Must be implemented by subclasses to calculate the gradient (first derivative) and Hessian (second
         * derivative) of the loss function for a certain example and label.
         *
         * @param labelMatrix       A pointer to an object of type `AbstractLabelMatrix` that provides random access to
                                    the labels of the training examples
         * @param exampleIndex      The index of the example for which the gradient and Hessian should be calculated
         * @param labelIndex        The index of the label for which the gradient and Hessian should be calculated
         * @param predictedScore    A scalar of type `float64`, representing the score that is predicted for the
         *                          respective example and label
         * @return                  A pair that contains two scalars of type `float64`, representing the gradient and
         *                          the Hessian that have been calculated
         */
        virtual std::pair<float64, float64> calculateGradientAndHessian(statistics::AbstractLabelMatrix* labelMatrix,
                                                                        intp exampleIndex, intp labelIndex,
                                                                        float64 predictedScore);

    };

}
