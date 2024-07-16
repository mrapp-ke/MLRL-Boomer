#include "mlrl/boosting/rule_evaluation/head_type_auto.hpp"

#include "mlrl/boosting/rule_evaluation/head_type_complete.hpp"
#include "mlrl/boosting/rule_evaluation/head_type_single.hpp"

namespace boosting {

    AutomaticHeadConfig::AutomaticHeadConfig(ReadableProperty<ILossConfig> lossConfigGetter,
                                             ReadableProperty<ILabelBinningConfig> labelBinningConfigGetter,
                                             ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter,
                                             ReadableProperty<IRegularizationConfig> l1RegularizationConfigGetter,
                                             ReadableProperty<IRegularizationConfig> l2RegularizationConfigGetter)
        : lossConfig_(lossConfigGetter), labelBinningConfig_(labelBinningConfigGetter),
          multiThreadingConfig_(multiThreadingConfigGetter), l1RegularizationConfig_(l1RegularizationConfigGetter),
          l2RegularizationConfig_(l2RegularizationConfigGetter) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory> AutomaticHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const IDecomposableClassificationLossConfig& lossConfig) const {
        if (labelMatrix.getNumOutputs() > 1) {
            SingleOutputHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                              l2RegularizationConfig_);
            return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
        } else {
            CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                          l2RegularizationConfig_);
            return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
        }
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory> AutomaticHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ISparseDecomposableClassificationLossConfig& lossConfig) const {
        if (labelMatrix.getNumOutputs() > 1) {
            SingleOutputHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                              l2RegularizationConfig_);
            return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
        } else {
            CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                          l2RegularizationConfig_);
            return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
        }
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory> AutomaticHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const INonDecomposableClassificationLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                      l2RegularizationConfig_);
        return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig, blas, lapack);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory> AutomaticHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
      const IDecomposableRegressionLossConfig& lossConfig) const {
        CompleteHeadConfig headConfig(labelBinningConfigGetter_, multiThreadingConfigGetter_,
                                      l1RegularizationConfigGetter_, l2RegularizationConfigGetter_);
        return headConfig.createStatisticsProviderFactory(featureMatrix, regressionMatrix, lossConfig);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory> AutomaticHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
      const INonDecomposableRegressionLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        CompleteHeadConfig headConfig(labelBinningConfigGetter_, multiThreadingConfigGetter_,
                                      l1RegularizationConfigGetter_, l2RegularizationConfigGetter_);
        return headConfig.createStatisticsProviderFactory(featureMatrix, regressionMatrix, lossConfig, blas, lapack);
    }

    bool AutomaticHeadConfig::isPartial() const {
        return lossConfig_.get().isDecomposable();
    }

    bool AutomaticHeadConfig::isSingleOutput() const {
        return lossConfig_.get().isDecomposable();
    }

}
