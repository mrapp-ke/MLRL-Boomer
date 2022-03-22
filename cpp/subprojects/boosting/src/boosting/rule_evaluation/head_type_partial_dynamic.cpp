#include "boosting/rule_evaluation/head_type_partial_dynamic.hpp"
#include "common/util/validation.hpp"


namespace boosting {

    DynamicPartialHeadConfig::DynamicPartialHeadConfig(
            const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr,
            const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
            const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : varianceThreshold_(0.1f), labelBinningConfigPtr_(labelBinningConfigPtr),
          multiThreadingConfigPtr_(multiThreadingConfigPtr), l1RegularizationConfigPtr_(l1RegularizationConfigPtr),
          l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    float32 DynamicPartialHeadConfig::getVarianceThreshold() const {
        return varianceThreshold_;
    }

    IDynamicPartialHeadConfig& DynamicPartialHeadConfig::setVarianceThreshold(float32 varianceThreshold) {
        assertGreater<float32>("varianceThreshold", varianceThreshold, 0);
        assertLess<float32>("varianceThreshold", varianceThreshold, 1);
        varianceThreshold_ = varianceThreshold;
        return *this;
    }

    std::unique_ptr<IStatisticsProviderFactory> DynamicPartialHeadConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
            const ILabelWiseLossConfig& lossConfig) const {
        // TODO
        return nullptr;
    }

    std::unique_ptr<IStatisticsProviderFactory> DynamicPartialHeadConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
            const IExampleWiseLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        // TODO
        return nullptr;
    }

}
