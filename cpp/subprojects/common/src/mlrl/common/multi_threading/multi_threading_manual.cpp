#include "mlrl/common/multi_threading/multi_threading_manual.hpp"

#include "mlrl/common/util/threads.hpp"
#include "mlrl/common/util/validation.hpp"

ManualMultiThreadingConfig::ManualMultiThreadingConfig() : numPreferredThreads_(0) {}

uint32 ManualMultiThreadingConfig::getNumPreferredThreads() const {
    return numPreferredThreads_;
}

IManualMultiThreadingConfig& ManualMultiThreadingConfig::setNumPreferredThreads(uint32 numPreferredThreads) {
    if (numPreferredThreads != 0) util::assertGreaterOrEqual<uint32>("numPreferredThreads", numPreferredThreads, 1);
    numPreferredThreads_ = numPreferredThreads;
    return *this;
}

MultiThreadingSettings ManualMultiThreadingConfig::getSettings(const IFeatureMatrix& featureMatrix,
                                                               uint32 numOutputs) const {
    return MultiThreadingSettings(util::getNumAvailableThreads(numPreferredThreads_));
}
