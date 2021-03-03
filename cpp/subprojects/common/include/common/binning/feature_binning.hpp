/**
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/feature_vector.hpp"
#include "common/binning/threshold_vector.hpp"
#include <memory>
#include <functional>

/**
 * Defines an interface for methods that assign feature values to bins.
 */
class IFeatureBinning {

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

        virtual ~IFeatureBinning() { };

        /**
         * A callback function that is invoked when a value is assigned to a bin. It takes the index of the bin, the
         * original index of the value, as well as the value itself, as arguments.
         */
        typedef std::function<void(uint32 binIndex, uint32 originalIndex, float32 value)> Callback;

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
         * @param callback      A callback that is invoked when a value is assigned to a bin
         * @return              An object of type `ThresholdVector` that stores the thresholds that result from the
         *                      boundaries between the bins
         */
        virtual std::unique_ptr<ThresholdVector> createBins(FeatureInfo featureInfo, const FeatureVector& featureVector,
                                                            Callback callback) const = 0;

};
