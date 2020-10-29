/**
 * Provides classes that implement strategies for sub-sampling training examples, features or labels.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"
#include "indices.h"
#include "random.h"
#include <memory>


/**
 * Defines an interface for one-dimensional vectors that provide access to weights.
 */
class IWeightVector : virtual public IRandomAccessVector<uint32> {

    public:

        virtual ~IWeightVector() { };

        /**
         * Returns whether the vector contains any zero weights or not.
         *
         * @return True, if the vector contains any zero weights, false otherwise
         */
        virtual bool hasZeroWeights() const = 0;

        /**
         * Returns the sum of the weights in the vector.
         *
         * @return The sum of the weights
         */
        virtual uint32 getSumOfWeights() const = 0;

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
         * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be
         *                      used
         * @return              An unique pointer to an object type `WeightVector` that provides access to the weights
         *                      of the individual training examples
         */
        virtual std::unique_ptr<IWeightVector> subSample(uint32 numExamples, RNG& rng) const = 0;

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

        std::unique_ptr<IWeightVector> subSample(uint32 numExamples, RNG& rng) const override;

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

        std::unique_ptr<IWeightVector> subSample(uint32 numExamples, RNG& rng) const override;

};

/**
 * An implementation of the class `IInstanceSubSampling` that does not perform any sampling, but assigns equal weights
 * to all examples.
 */
class NoInstanceSubSamplingImpl : virtual public IInstanceSubSampling {

    public:

        std::unique_ptr<IWeightVector> subSample(uint32 numExamples, RNG& rng) const override;

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
         * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be
         *                      used
         * @return              An unique pointer to an object of type `IIndexVector` that provides access to the
         *                      indices of the features that are contained in the sub-sample
         */
        virtual std::unique_ptr<IIndexVector> subSample(uint32 numFeatures, RNG& rng) const = 0;

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

        std::unique_ptr<IIndexVector> subSample(uint32 numFeatures, RNG& rng) const override;

};

/**
 * An implementation of the class `IFeatureSubSampling` that does not perform any sampling, but includes all features.
 */
class NoFeatureSubSamplingImpl : virtual public IFeatureSubSampling {

    public:

        std::unique_ptr<IIndexVector> subSample(uint32 numFeatures, RNG& rng) const override;

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
         * @param rng       A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return          An unique pointer to an object of type `IIndexVector` that provides access to the indices of
         *                  the labels that are contained in the sub-sample
         */
        virtual std::unique_ptr<IIndexVector> subSample(uint32 numLabels, RNG& rng) const = 0;

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

        std::unique_ptr<IIndexVector> subSample(uint32 numLabels, RNG& rng) const override;

};

/**
 * An implementation of the class `ILabelSubSampling` that does not perform any sampling, but includes all labels.
 */
class NoLabelSubSamplingImpl : virtual public ILabelSubSampling {

    public:

        std::unique_ptr<IIndexVector> subSample(uint32 numLabels, RNG& rng) const override;

};
