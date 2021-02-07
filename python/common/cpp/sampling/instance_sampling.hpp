/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "weight_vector.hpp"
#include "random.hpp"
#include <memory>


/**
 * Defines an interface for all classes that implement a strategy for sub-sampling training examples.
 */
class IInstanceSubSampling {

    public:

        virtual ~IInstanceSubSampling() { };

        /**
         * Creates and returns a sub-sample of the available training examples.
         *
         * @param numExamples   The total number of available training examples
         * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be
         *                      used
         * @return              An unique pointer to an object type `WeightVector` that provides access to the weights
         *                      of the individual training examples
         */
        virtual std::unique_ptr<IWeightVector> subSample(uint32 numExamples, RNG& rng) const = 0;

};
