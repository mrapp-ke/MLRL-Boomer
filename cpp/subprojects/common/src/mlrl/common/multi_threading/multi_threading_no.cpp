#include "mlrl/common/multi_threading/multi_threading_no.hpp"

uint32 NoMultiThreadingConfig::getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numOutputs) const {
    return 1;
}

MultiThreadingSettings NoMultiThreadingConfig::getSettings(const IFeatureMatrix& featureMatrix,
                                                           uint32 numOutputs) const {
    return MultiThreadingSettings(1);
}
