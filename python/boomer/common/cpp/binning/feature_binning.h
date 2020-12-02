/**
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../input/feature_vector.h"
#include "binning_observer.h"


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
         * @param observer      A reference to an object of type `IBinningObserver` that should be notified when a value
         *                      is assigned to a bin
         */
        virtual void createBins(FeatureInfo featureInfo, const FeatureVector& featureVector,
                                IBinningObserver<float32>& observer) const = 0;

};
