/**
 * Implements classes that provide access to approximate thresholds that may be used by the conditions of rules.
 *
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "thresholds.h"
#include <unordered_map>


/**
 * Provides access to the thresholds that result from applying a binning method to the feature values of the training
 * examples.
 */
class ApproximateThresholds : public AbstractThresholds {

    private:

        // Forward declarations
        class ThresholdsSubset;

        /**
         * A wrapper for statistics and bins that is stored in the cache.
         */
        struct BinCacheEntry {
            std::unique_ptr<IStatistics> statisticsPtr;
            std::unique_ptr<BinVector> binVectorPtr;
        };

        std::shared_ptr<IBinning> binningPtr_;

        std::unordered_map<uint32, BinCacheEntry> cache_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureMaskPtr     A shared pointer to an object of type `INominalFeatureMask` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsPtr             A shared pointer to an object of type `IStatistics` that provides access to
         *                                  statistics about the labels of the training examples
         * @param headRefinementFactoryPtr  A shared pointer to an object of type `IHeadRefinementFactory` that allows
         *                                  to create instances of the class that should be used to find the heads of
         *                                  rules
         * @param binningPtr                A shared pointer to an object of type `IBinning` that implements the binning
         *                                  method to be used
         */
        ApproximateThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                              std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
                              std::shared_ptr<IStatistics> statisticsPtr,
                              std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
                              std::shared_ptr<IBinning> binningPtr);

        std::unique_ptr<IThresholdsSubset> createSubset(const IWeightVector& weights) override;

};
