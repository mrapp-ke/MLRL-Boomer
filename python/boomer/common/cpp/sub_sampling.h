/**
 * Provides classes that implement strategies for sub-sampling training examples, features or labels.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "data.h"
#include "random.h"


/**
 * Defines an interface for one-dimensional, potentially sparse, vectors that provide access to weights.
 */
class IWeightVector : virtual public ISparseRandomAccessVector<uint32> {

    public:

        virtual ~IWeightVector() { };

        /**
         * Returns the sum of the weights in the vector.
         *
         * @return The sum of the weights
         */
        virtual uint32 getSumOfWeights() = 0;

};

/**
 * An one-dimensional vector that provides access to weights that are stored in a C-contiguous array.
 */
template<class T>
class DenseWeightVector : virtual public IWeightVector {

    private:

        DenseVector<T>* weights_;

        uint32 sumOfWeights_;

    public:

        /**
         * @param weights       A pointer to an object of type `DenseVector<T>` that stores the weights
         * @param sumOfWeights  The sum of the weights in the given vector
         */
        DenseWeightVector(DenseVector<T>* weights, uint32 sumOfWeights);

        ~DenseWeightVector();

        uint32 getNumElements() override;

        bool hasZeroElements() override;

        uint32 getValue(uint32 pos) override;

        uint32 getSumOfWeights() override;

};

/**
 * An one-dimensional that provides access to equal weights.
 */
class EqualWeightVector : virtual public IWeightVector {

    private:

        uint32 numElements_;

    public:

        /**
         * @param numTotalElements The number of elements in the vector. Must be at least 1
         */
        EqualWeightVector(uint32 numElements);

        uint32 getNumElements() override;

        bool hasZeroElements();

        uint32 getValue(uint32 pos) override;

        uint32 getSumOfWeights() override;

};

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
         * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be
         *                      used
         * @return              A pointer to an object type `WeightVector`, shape `(numExamples)`, that provides access
         *                      to the weights of the individual training examples, i.e., how many times each of the
         *                      examples is contained in the sample, as well as the sum of the weights
         */
        virtual IWeightVector* subSample(uint32 numExamples, RNG* rng) = 0;

};

/**
 * An implementation of the class `IInstanceSubSampling` that does not perform any sampling, but assigns equal weights
 * to all examples.
 */
class NoInstanceSubSamplingImpl : virtual public IInstanceSubSampling {

    public:

        IWeightVector* subSample(uint32 numExamples, RNG* rng) override;

};

/**
 * Defines an interface for all classes that implement a strategy for sub-sampling features.
 */
class IFeatureSubSampling {

    public:

        virtual ~IFeatureSubSampling() { };

        /**
         * Creates and returns a sub-sample of the available features.
         *
         * @param numFeatures   The total number of available features
         * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be
         *                      used
         * @return              A pointer to an object of type `IIndexVector`, shape `(numSamples)`, that provides
         *                      access to the indices of the features that are contained in the sub-sample
         */
        virtual IIndexVector* subSample(uint32 numFeatures, RNG* rng) = 0;

};

/**
 * An implementation of the class `IFeatureSubSampling` that does not perform any sampling, but includes all features.
 */
class NoFeatureSubSamplingImpl : virtual public IFeatureSubSampling {

    public:

        IIndexVector* subSample(uint32 numFeatures, RNG* rng) override;

};

/**
 * Defines an interface for all classes that implement a strategy for sub-sampling labels.
 */
class ILabelSubSampling {

    public:

        virtual ~ILabelSubSampling() { };

        /**
         * Creates and returns a sub-sample of the available labels.
         *
         * @param numLabels The total number of available labels
         * @param rng       A pointer to an object of type `RNG`, implementing the random number generator to be used
         * @return          A pointer to an object of type `IIndexVector`, shape `(numSamples)`, that provides access to
         *                  the indices of the labels that are contained in the sub-sample
         */
        virtual IIndexVector* subSample(uint32 numLabels, RNG* rng) = 0;

};

/**
 * An implementation of the class `ILabelSubSampling` that does not perform any sampling, but includes all labels.
 */
class NoLabelSubSamplingImpl : virtual public ILabelSubSampling {

    public:

        IIndexVector* subSample(uint32 numLabels, RNG* rng) override;

};
