/**
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/feature_vector.hpp"
#include "common/binning/threshold_vector.hpp"
#include <memory>


/**
 * A vector that stores the indices of the bins, individual examples belong to.
 */
typedef DenseVector<uint32> BinIndexVector;

/**
 * Defines an interface for methods that assign feature values to bins.
 */
class IFeatureBinning {

    public:

        /**
         * The result that is returned by a binning method. It contains an unique pointer to a vector that stores the
         * thresholds that result from the boundaries of the bins, as well as to a vector that stores the indices of the
         * bins, individual values have been assigned to.
         */
        struct Result {
            std::unique_ptr<ThresholdVector> thresholdVectorPtr;
            std::unique_ptr<BinIndexVector> binIndicesPtr;
        };

        virtual ~IFeatureBinning() { };

        /**
         * Assigns the values in a given `FeatureVector` to bins.
         *
         * @param featureVector A reference to an object of type `FeatureVector` whose values should be assigned to bins
         * @return              An object of type `Result` that contains a vector, which stores thresholds that result
         *                      from the boundaries between the bins, as well as a vector that stores the indices of the
         *                      bins, individual values have been assigned to
         */
        virtual Result createBins(FeatureVector& featureVector) const = 0;

};
