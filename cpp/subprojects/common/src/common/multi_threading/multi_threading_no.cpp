#include "common/multi_threading/multi_threading_no.hpp"


uint32 NoMultiThreadingConfig::getNumThreads(const IFeatureMatrix& featureMatrix,
                                             const ILabelMatrix& labelMatrix) const {
    return 1;
}
