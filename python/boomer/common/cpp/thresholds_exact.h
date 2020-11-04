/**
 * Implements classes that provide access to exact thresholds that may be used by the conditions of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "thresholds.h"
#include <unordered_map>


/**
 * Provides access to all thresholds that result from the feature values of the training examples.
 */
class ExactThresholds : public AbstractThresholds {

    private:

        // Forward declarations
        class ThresholdsSubset;

        std::unordered_map<uint32, std::unique_ptr<FeatureVector>> cache_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureMaskPtr     A shared pointer to an object of type `INominalFeatureMask` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsPtr             A shared pointer to an object of type `AbstractStatistics` that provides
         *                                  access to statistics about the labels of the training examples
         * @param headRefinementFactoryPtr  A shared pointer to an object of type `IHeadRefinementFactory` that allows
         *                                  to create instances of the class that should be used to find the heads of
         *                                  rules
         */
        ExactThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                        std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
                        std::shared_ptr<AbstractStatistics> statisticsPtr,
                        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr);

        std::unique_ptr<IThresholdsSubset> createSubset(const IWeightVector& weights) override;

};
