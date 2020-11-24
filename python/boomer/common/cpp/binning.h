/**
 * Implements different methods for assigning floating point values to bins.
 *
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "input/feature_vector.h"


/**
 * Defines an interface to be implemented by classes that should be notified when values are assigned to bins.
 *
 * @tparam T The type of the values that are assigned to bins
 */
template<class T>
class IBinningObserver {

    public:

        virtual ~IBinningObserver() { };

        /**
         * Notifies the observer that a value has been assigned to a certain bin.
         *
         * @param binIndex      The index of the bin, the value is assigned to
         * @param originalIndex The original index of the value
         * @param value         The value
         */
        virtual void onBinUpdate(uint32 binIndex, uint32 originalIndex, T value) = 0;

};

/**
 * Defines an interface for methods that assign floating point values to bins.
 */
class IBinning {

    public:

        /**
         * Stores information about the values in a `FeatureVector`. This includes the number of bins, the values should
         * be assigned to, as well as the minimum and maximum value in the vector.
         */
        struct FeatureInfo {
            uint32 numBins;
            float32 minValue;
            float32 maxValue;
        };

        virtual ~IBinning() { };

        /**
         * Retrieves and returns information about the values in a given `FeatureVector` that is required to apply the
         * binning method.
         *
         * This function must be called prior to the function `createBins` to obtain information, e.g. the number of
         * bins to be used, that is required to apply the binning method. This function may also be used to prepare,
         * e.g. sort, the given `FeatureVector`. The `FeatureInfo` returned by this function must be passed to the
         * function `createBins` later on.
         *
         * @param featureVector A reference to an object of type `FeatureVector` whose values should be assigned to bins
         * @return              A struct of `type `FeatureInfo` that stores the information
         */
        virtual FeatureInfo getFeatureInfo(FeatureVector& featureVector) const = 0;

        /**
         * Assigns the values in a given `FeatureVector` to bins.
         *
         * @param featureInfo   A struct of type `FeatureInfo` that stores information about the given `FeatureVector`
         * @param featureVector A reference to an object of type `FeatureVector` whose values should be assigned to bins
         * @param observer      A reference to an object of type `IBinningObserver`, which should be notified when a
         *                      value is assigned to a bin
         */
        virtual void createBins(FeatureInfo featureInfo, const FeatureVector& featureVector,
                                IBinningObserver<float32>& observer) const = 0;

};

/**
 * Assigns floating point values to bins in a way such that each bin contains approximately the same number of values.
 */
class EqualFrequencyBinningImpl : public IBinning {

    private:

        float32 binRatio_;

    public:

        /**
         * @param binRatio A percentage that specifies how many bins should be used to assign the values in an array to,
         *                 e.g., if 100 values are available, 0.5 means that `ceil(0.5 * 100) = 50` bins should be used
         */
        EqualFrequencyBinningImpl(float32 binRatio);

        FeatureInfo getFeatureInfo(FeatureVector& featureVector) const override;

        void createBins(FeatureInfo featureInfo, const FeatureVector& featureVector,
                        IBinningObserver<float32>& observer) const override;

};

/**
 * Assigns floating point values to bins in a way such that each bin contains values from equally sized value ranges.
 */
class EqualWidthBinningImpl : public IBinning {

    private:

        float32 binRatio_;

    public:

        /**
         * @param binRatio A percentage that specifies how many bins should be used to assign the values in an array to,
         *                 e.g., if 100 values are available, 0.5 means that `ceil(0.5 * 100) = 50` bins should be used
         */
        EqualWidthBinningImpl(float32 binRatio);

        FeatureInfo getFeatureInfo(FeatureVector& featureVector) const override;

        void createBins(FeatureInfo featureInfo, const FeatureVector& featureVector,
                        IBinningObserver<float32>& observer) const override;

};
