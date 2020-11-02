/**
 * Implements different methods for assigning floating point values to bins.
 *
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "input_data.h"


/**
 * Defines an interface to be implemented by classes that should be notified when values are assigned to bins.
 */
class IBinningObserver {

    public:

        virtual ~IBinningObserver() { };

        /**
         * Notifies the observer that a value has been assigned to a certain bin.
         *
         * @param binIndex  The index of the bin, the value is assigned to
         * @param entry     A reference to a struct of type `FeatureVector::Entry` that contains the value and its index
         *                  in the original array
         */
        virtual void onBinUpdate(uint32 binIndex, const FeatureVector::Entry& entry) = 0;

};

/**
 * Defines an interface for methods that assign floating point values to bins.
 */
class IBinning {

    public:

        virtual ~IBinning() { };

        virtual uint32 getNumBins(FeatureVector& featureVector) const = 0;

        /**
         * Assigns the values in an array to bins.
         *
         * @param numBins       The number of bins to be used
         * @param featureVector A reference to an object of type `FeatureVector` whose values should be assigned to the
         *                      bins
         * @param observer      A reference to an object of type `IBinningObserver`, which should be notified when a
         *                      value is assigned to a bin
         */
        virtual void createBins(uint32 numBins, FeatureVector& featureVector, IBinningObserver& observer) = 0;

};

/**
 * Assigns floating point values to bins in a way such that each bin contains approximately the same number of values.
 */
class EqualFrequencyBinningImpl : virtual public IBinning {

    private:

        float32 binRatio_;

    public:

        EqualFrequencyBinningImpl(float32 binRatio);

        uint32 getNumBins(FeatureVector& featureVector) const override;

        void createBins(uint32 numBins, FeatureVector& featureVector, IBinningObserver& observer) override;

};

/**
 * Assigns floating point values to bins in a way such that each bin contains values from equally sized value ranges.
 */
class EqualWidthBinningImpl : virtual public IBinning {

    private:

        float32 binRatio_;

    public:

        EqualWidthBinningImpl(float32 binRatio);

        uint32 getNumBins(FeatureVector& featureVector) const override;

        void createBins(uint32 numBins, FeatureVector& featureVector, IBinningObserver& observer) override;

};
