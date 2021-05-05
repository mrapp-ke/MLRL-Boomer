/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/indices/index_vector.hpp"
#include "common/sampling/random.hpp"
#include <memory>


/**
 * Defines an interface for all classes that implement a strategy for sub-sampling features.
 */
class IFeatureSubSampling {

    public:

        virtual ~IFeatureSubSampling() { };

        /**
         * Creates and returns a sub-sample of the available features.
         *
         * @param rng   A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return      A reference to an object of type `IIndexVector` that provides access to the indices of the
         *              features that are contained in the sub-sample
         */
        virtual const IIndexVector& subSample(RNG& rng) = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IFeatureSubSampling`.
 */
class IFeatureSubSamplingFactory {

    public:

        virtual ~IFeatureSubSamplingFactory() { };

        /**
         * Creates and returns a new object of type `IFeatureSubSampling`.
         *
         * @param numFeatures   The total number of available features
         * @return              An unique pointer to an object of type `IFeatureSubSampling` that has been created
         */
        virtual std::unique_ptr<IFeatureSubSampling> create(uint32 numFeatures) const = 0;

};
