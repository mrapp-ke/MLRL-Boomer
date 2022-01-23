#include "common/multi_threading/multi_threading_no.hpp"


uint32 NoMultiThreadingConfig::configure(const ILabelMatrix& labelMatrix) const {
    return 1;
}
