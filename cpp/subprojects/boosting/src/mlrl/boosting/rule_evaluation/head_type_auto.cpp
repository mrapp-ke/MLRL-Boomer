#include "mlrl/boosting/rule_evaluation/head_type_auto.hpp"

#include "mlrl/boosting/rule_evaluation/head_type_complete.hpp"
#include "mlrl/boosting/rule_evaluation/head_type_single.hpp"

namespace boosting {

    AutomaticHeadConfig::AutomaticHeadConfig(ReadableProperty<ILossConfig> lossConfig,
                                             ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                             ReadableProperty<IMultiThreadingConfig> multiThreadingConfig,
                                             ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                                             ReadableProperty<IRegularizationConfig> l2RegularizationConfig)
        : lossConfig_(lossConfig), labelBinningConfig_(labelBinningConfig), multiThreadingConfig_(multiThreadingConfig),
          l1RegularizationConfig_(l1RegularizationConfig), l2RegularizationConfig_(l2RegularizationConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      AutomaticHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const IDecomposableClassificationLossConfig& lossConfig) const {
        CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                      l2RegularizationConfig_);
        return headConfig.createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      AutomaticHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const ISparseDecomposableClassificationLossConfig& lossConfig) const {
        if (labelMatrix.getNumOutputs() > 1) {
            SingleOutputHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                              l2RegularizationConfig_);
            return headConfig.createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
        } else {
            CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                          l2RegularizationConfig_);
            return headConfig.createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
        }
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      AutomaticHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const INonDecomposableClassificationLossConfig& lossConfig, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory) const {
        CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                      l2RegularizationConfig_);
        return headConfig.createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig,
                                                                        blasFactory, lapackFactory);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      AutomaticHeadConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const IDecomposableRegressionLossConfig& lossConfig) const {
        CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                      l2RegularizationConfig_);
        return headConfig.createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix, lossConfig);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      AutomaticHeadConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const INonDecomposableRegressionLossConfig& lossConfig, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory) const {
        CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                      l2RegularizationConfig_);
        return headConfig.createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix, lossConfig,
                                                                    blasFactory, lapackFactory);
    }

    bool AutomaticHeadConfig::isPartial() const {
        return lossConfig_.get().isDecomposable() && lossConfig_.get().isSparse();
    }

    bool AutomaticHeadConfig::isSingleOutput() const {
        return lossConfig_.get().isDecomposable() && lossConfig_.get().isSparse();
    }

}
