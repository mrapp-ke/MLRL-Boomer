#include "mlrl/common/multi_threading/multi_threading_manual.hpp"

#include "mlrl/common/util/threads.hpp"
#include "mlrl/common/util/validation.hpp"

ManualMultiThreadingConfig::ManualMultiThreadingConfig() : numPreferredThreads_(0) {}

uint32 ManualMultiThreadingConfig::getNumPreferredThreads() const {
    return numPreferredThreads_;
}

IManualMultiThreadingConfig& ManualMultiThreadingConfig::setNumPreferredThreads(uint32 numPreferredThreads) {
    if (numPreferredThreads != 0) assertGreaterOrEqual<uint32>("numPreferredThreads", numPreferredThreads, 1);
    numPreferredThreads_ = numPreferredThreads;
    return *this;
}

uint32 ManualMultiThreadingConfig::getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return getNumAvailableThreads(numPreferredThreads_);
}
