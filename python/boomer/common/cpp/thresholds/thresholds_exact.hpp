/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "thresholds_factory.hpp"


/**
 * A factory that allows to create instances of the type `ExactThresholds`.
 */
class ExactThresholdsFactory final : public IThresholdsFactory {

    public:

        std::unique_ptr<IThresholds> create(
            std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
            std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
            std::shared_ptr<IStatisticsProvider> statisticsProviderPtr,
            std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr) const override;

};
