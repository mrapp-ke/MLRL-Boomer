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

        T* weights_;

        uint32 numElements_;

        uint32 sumOfWeights_;

    public:

        /**
         * @param weights       A pointer to an array of template type `T`, shape `(numElements)`, that stores the
         *                      weights
         * @param numElements   The number of elements in the vector. Must be at least 1
         * @param sumOfWeights  The sum of the weights in the vector
         */
        DenseWeightVector(T* weights, uint32 numElements, uint32 sumOfWeights);

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
 * An one-dimensional vector that provides random access to a fixed number of indices stored in a C-contiguous array.
 */
class DenseIndexVector : virtual public IIndexVector {

    private:

        uint32 numElements_;

        uint32* indices_;

    public:

        /**
         * @param numElements The number of elements in the vector. Must be at least 1
         */
        DenseIndexVector(uint32 numElements);

        ~DenseIndexVector();

        /**
         * Sets the index at a specific position.
         *
         * @param pos   The position of the index. Must be in [0, getNumElements())
         * @param index The index to be set. Must be at least 0
         */
        void setIndex(uint32 pos, uint32 index);

        uint32 getNumElements() override;

        bool hasZeroElements() override;

        uint32 getIndex(uint32 pos) override;

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
 * Implements bootstrap aggregating (bagging) for selecting a subset of the available training examples with
 * replacement.
 */
class BaggingImpl : virtual public IInstanceSubSampling {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1]
         */
        BaggingImpl(float32 sampleSize);

        IWeightVector* subSample(uint32 numExamples, RNG* rng) override;

};

/**
 * Implements random instance subset selection for selecting a subset of the available training examples without
 * replacement.
 */
class RandomInstanceSubsetSelectionImpl : virtual public IInstanceSubSampling {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1)
         */
        RandomInstanceSubsetSelectionImpl(float32 sampleSize);

        IWeightVector* subSample(uint32 numExamples, RNG* rng) override;

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
 * Implements random feature subset selection for selecting a random subset of the available features without
 * replacement.
 */
class RandomFeatureSubsetSelectionImpl : virtual public IFeatureSubSampling {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available features). Must be in (0, 1) or 0, if the default sample size
         *                   `floor(log2(num_features - 1) + 1)` should be used
         */
        RandomFeatureSubsetSelectionImpl(float32 sampleSize);

        IIndexVector* subSample(uint32 numFeatures, RNG* rng) override;

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
 * Implements random label subset selection for selecting a random subset of the available features without replacement.
 */
class RandomLabelSubsetSelectionImpl : virtual public ILabelSubSampling {

    private:

        uint32 numSamples_;

    public:

        /**
         * @param The number of labels to be included in the sample
         */
        RandomLabelSubsetSelectionImpl(uint32 numSamples);

        IIndexVector* subSample(uint32 numLabels, RNG* rng) override;

};

/**
 * An implementation of the class `ILabelSubSampling` that does not perform any sampling, but includes all labels.
 */
class NoLabelSubSamplingImpl : virtual public ILabelSubSampling {

    public:

        IIndexVector* subSample(uint32 numLabels, RNG* rng) override;

};
