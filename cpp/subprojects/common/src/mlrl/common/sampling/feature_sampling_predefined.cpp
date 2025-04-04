#include "mlrl/common/sampling/feature_sampling_predefined.hpp"

PredefinedFeatureSampling::PredefinedFeatureSampling(const IIndexVector& indexVector) : indexVector_(indexVector) {}

const IIndexVector& PredefinedFeatureSampling::sample() {
    return indexVector_;
}

std::unique_ptr<IFeatureSampling> PredefinedFeatureSampling::createBeamSearchFeatureSampling(bool resample) {
    return std::make_unique<PredefinedFeatureSampling>(indexVector_);
}
