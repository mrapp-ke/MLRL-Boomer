/**
 * Implements different methods for assigning floating point values to bins.
 *
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "statistics.h"
#include "tuples.h"


/**
 * Defines an interface to be implemented by classes that should be notified when values are assigned to bins.
 */
class IBinningObserver {

    public:

        virtual ~IBinningObserver() { };

        /**
         * Will be called everytime a value is assigned to a bin.
         *
         * @param binIndex      Index of the bin the value is assigned to
         * @param indexedValue  The value and corresponding index which is assigned
         */
        virtual void onBinUpdate(uint32 binIndex, IndexedFloat32* indexedValue) = 0;

};

/**
 * Defines an interface for methods that assign floating point values to bins.
 */
class IBinning {

    public:

        virtual ~IBinning() { };

        /**
         * Must be implemented by subclasses. Create a number of bags and assigns values to those bags. The results will
         * be passed to an observer which handles the bag management.
         *
         * @param numBins       The number of bins which should be considered
         * @param indexedArray  An array of examples, which should be put in the bins
         * @param observer      The `IBinningObserver` who is notified, when new results are available
         */
        virtual void createBins(uint32 numBins, IndexedFloat32Array* indexedArray, IBinningObserver* observer) = 0;

};

/**
 *
 */
class IHistogramBuilder : virtual public IBinningObserver {

    public:

        virtual AbstractStatistics* buildHistogram() = 0;

};

/**
 * Assigns floating point values to bins in a way such that each bin contains approximately the same number of values.
 */
class EqualFrequencyBinningImpl : virtual public IBinning {

    public:

        void createBins(uint32 numBins, IndexedFloat32Array* indexedArray, IBinningObserver* observer) override;

};

/**
 * Assigns floating point values to bins in a way such that each bin contains values from equally sized value ranges.
 */
class EqualWidthBinningImpl : virtual public IBinning {

    public:

        void createBins(uint32 numBins, IndexedFloat32Array* indexedArray, IBinningObserver* observer) override;

};

/**
 *
 */
class DenseLabelWiseStatisticsImpl : virtual public IHistogramBuilder {

    private:

        DenseLabelWiseStatisticsImpl* statistics_;

        uint32 numBins;

        float64* gradients_;

        float64* hessians_;

    public:

        DenseLabelWiseStatisticsImpl(uint32 numBins);

        void HistogramBuilderImpl(DenseLabelWiseStatisticsImpl* statistics, uint32 numBins);

        void onBinUpdate(uint32 binIndex, IndexedFloat32* indexedValue) override;

        AbstractStatistics* build() override;

};

/**
 *
 */
class DenseExampleWiseStatisticsImpl : virtual public IHistogramBuilder {

    private:

        DenseExampleWiseStatisticsImpl* statistics_;

        uint32 numBins;

        float64* gradients_;

        float64* hessians_;

    public:

        DenseExampleWiseStatisticsImpl(uint32 numBins);

        void HistogramBuilderImpl(DenseExampleWiseStatisticsImpl* statistics, uint32 numBins);

        void onBinUpdate(uint32 binIndex, IndexedFloat32* indexedValue) override;

        AbstractStatistics* build() override;

};
