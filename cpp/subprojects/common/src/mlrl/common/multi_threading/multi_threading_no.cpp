#include "mlrl/common/multi_threading/multi_threading_no.hpp"

MultiThreadingSettings NoMultiThreadingConfig::getSettings(const IFeatureMatrix& featureMatrix,
                                                           uint32 numOutputs) const {
    return MultiThreadingSettings(1);
}
