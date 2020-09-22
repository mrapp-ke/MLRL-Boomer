/**
 * Implements the unsupervised binning methods on examples
 *
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */

#pragma once

#include "arrays.h"
#include "statistics.h"
#include "tuples.h"

class IBinningObserver{

    public:

        virtual ~IBinningObserver() { };

        /**
        *   Will be called everytime an example is assigned to a bin.
        *
        *   @param binIndex         Index of the bin the example is assigned to
        *   @param indexedValue     The example value which is assigned
        */
        virtual void onBinUpdate(intp binIndex, IndexedFloat32* indexedValue) = 0;

};

class IBinning{

    public:

        virtual ~IBinning() { };

        /**
        *   Must be implemented by subclasses. Create a number of bags and assigns examples to those bags. The results
        *   will be passed to a observer which handles the bag management.
        *
        *   @param numBins          The number of bins which should be considered
        *   @param indexedArray     An array of examples, which should be put in the bins
        *   @param observer         The `IBinningObserver` who is notified, when new results are available
        */
        virtual void createBins(uint32 numBins, IndexedFloat32Array* indexedArray, IBinningObserver* observer) = 0;

};

class EqualFrequencyBinningImpl : virtual public IBinning{

    public:

        /**
        *   Create a number of bags and assigns examples to those bags. The results will be passed to a observer which
        *   handles the bag management. Each bin will get a fix number of examples with minor differences due to
        *   divisibility and equality of examples
        *
        *   @param numBins          The number of bins which should be considered
        *   @param indexedArray     An array of examples, which should be put in the bins
        *   @param observer         The `IBinningObserver` who is notified, when new results are available
        */
        void createBins(uint32 numBins, IndexedFloat32Array* indexedArray, IBinningObserver* observer) override;

};

class EqualWidthBinningImpl : virtual public IBinning{

    public:

        /**
        *   Create a number of bags and assigns examples to those bags. The results will be passed to a observer which
        *   handles the bag management. Each bin will get examples in a certain range of values
        *
        *   @param numBins          The number of bins which should be considered
        *   @param indexedArray     An array of examples, which should be put in the bins
        *   @param observer         The `IBinningObserver` who is notified, when new results are available
        */
        void createBins(uint32 numBins, IndexedFloat32Array* indexedArray, IBinningObserver* observer) override;

};