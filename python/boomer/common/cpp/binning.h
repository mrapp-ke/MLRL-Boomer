/**
 * Implements different methods for assigning floating point values to bins.
 *
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "tuples.h"


/**
 * Defines an interface to be implemented by classes that should be notified when values are assigned to bins.
 */
class IBinningObserver {

    public:

        virtual ~IBinningObserver() { };

        /**
         * Notifies the observer that a value has been assigned to a certain bin.
         *
         * @param binIndex      The index of the bin, the value is assigned to
         * @param indexedValue  A reference to a struct of type `IndexedFloat32` that contains the value and its index
         *                      in the original array
         */
        virtual void onBinUpdate(uint32 binIndex, IndexedFloat32& indexedValue) = 0;

};

/**
 * Defines an interface for methods that assign floating point values to bins.
 */
class IBinning {

    public:

        virtual ~IBinning() { };

        /**
         * Assigns the values in an array to bins.
         *
         * @param numBins       The number of bins to be used
         * @param indexedArray  A reference to a struct of type `IndexedFloat32Array` that stores a pointer to an array
         *                      whose values should be assigned to the bins, as well as the number of elements in the
         *                      array
         * @param observer      A reference to an object of type `IBinningObserver`, which should be notified when a
         *                      value is assigned to a bin
         */
        virtual void createBins(uint32 numBins, IndexedFloat32Array& indexedArray, IBinningObserver& observer) = 0;

};

/**
 * Assigns floating point values to bins in a way such that each bin contains approximately the same number of values.
 */
class EqualFrequencyBinningImpl : virtual public IBinning {

    public:

        void createBins(uint32 numBins, IndexedFloat32Array& indexedArray, IBinningObserver& observer) override;

};

/**
 * Assigns floating point values to bins in a way such that each bin contains values from equally sized value ranges.
 */
class EqualWidthBinningImpl : virtual public IBinning {

    public:

        void createBins(uint32 numBins, IndexedFloat32Array& indexedArray, IBinningObserver& observer) override;

};
