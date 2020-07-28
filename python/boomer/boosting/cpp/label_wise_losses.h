/**
 * Implements different differentiable loss functions.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
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
         * @param exampleIndex      The index of the example for which the gradient and Hessian should be calculated
         * @param labelIndex        The index of the label for which the gradient and Hessian should be calculated
         * @param predictedScore    A scalar of type `float64`, representing the score that is predicted for the
         *                          respective example and label
         * @return                  A pair that contains two scalars of type `float64`, representing the gradient and
         *                          the Hessian that have been calculated
         */
        virtual std::pair<float64, float64> calculateGradientAndHessian(intp exampleIndex, intp labelIndex,
                                                                        float64 predictedScore);

    };

}
