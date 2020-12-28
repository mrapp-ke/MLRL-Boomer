/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/types.h"


/**
 * Defines an interface for all classes that represent the head of a rule.
 */
class IHead {

    public:

        virtual ~IHead { };

        /**
         * Adds the scores that are contained by the head to a given array of predictions.
         *
         * Optionally, a mask may be provided in order to restrict the prediction to certain labels.
         *
         * @param predictions   A pointer to an array of type `float64`, shape `(num_labels)`, which stores the
         *                      predictions to be updated
         * @param mask          A pointer to an array of type `uint8`, shape `(num_labels)`, that indicates for which
         *                      labels the head should predict or a null pointer, if the prediction should not be
         *                      restricted
         */
        virtual void apply(float64* predictions, uint8* mask) const = 0;

};
