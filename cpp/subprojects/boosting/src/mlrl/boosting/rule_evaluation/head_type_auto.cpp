#include "mlrl/boosting/rule_evaluation/head_type_auto.hpp"

#include "mlrl/boosting/rule_evaluation/head_type_complete.hpp"
#include "mlrl/boosting/rule_evaluation/head_type_single.hpp"

namespace boosting {

    AutomaticHeadConfig::AutomaticHeadConfig(GetterFunction<ILossConfig> lossConfigGetter,
                                             GetterFunction<ILabelBinningConfig> labelBinningConfigGetter,
                                             GetterFunction<IMultiThreadingConfig> multiThreadingConfigGetter,
                                             GetterFunction<IRegularizationConfig> l1RegularizationConfigGetter,
                                             GetterFunction<IRegularizationConfig> l2RegularizationConfigGetter)
        : lossConfigGetter_(lossConfigGetter), labelBinningConfigGetter_(labelBinningConfigGetter),
          multiThreadingConfigGetter_(multiThreadingConfigGetter),
          l1RegularizationConfigGetter_(l1RegularizationConfigGetter),
          l2RegularizationConfigGetter_(l2RegularizationConfigGetter) {}

    std::unique_ptr<IStatisticsProviderFactory> AutomaticHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const IDecomposableLossConfig& lossConfig) const {
        if (labelMatrix.getNumOutputs() > 1) {
            SingleOutputHeadConfig headConfig(labelBinningConfigGetter_, multiThreadingConfigGetter_,
                                              l1RegularizationConfigGetter_, l2RegularizationConfigGetter_);
            return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
        } else {
            CompleteHeadConfig headConfig(labelBinningConfigGetter_, multiThreadingConfigGetter_,
                                          l1RegularizationConfigGetter_, l2RegularizationConfigGetter_);
            return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
        }
    }

    std::unique_ptr<IStatisticsProviderFactory> AutomaticHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ISparseDecomposableLossConfig& lossConfig) const {
        if (labelMatrix.getNumOutputs() > 1) {
            SingleOutputHeadConfig headConfig(labelBinningConfigGetter_, multiThreadingConfigGetter_,
                                              l1RegularizationConfigGetter_, l2RegularizationConfigGetter_);
            return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
        } else {
            CompleteHeadConfig headConfig(labelBinningConfigGetter_, multiThreadingConfigGetter_,
                                          l1RegularizationConfigGetter_, l2RegularizationConfigGetter_);
            return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
        }
    }

    std::unique_ptr<IStatisticsProviderFactory> AutomaticHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const INonDecomposableLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        CompleteHeadConfig headConfig(labelBinningConfigGetter_, multiThreadingConfigGetter_,
                                      l1RegularizationConfigGetter_, l2RegularizationConfigGetter_);
        return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig, blas, lapack);
    }

    bool AutomaticHeadConfig::isPartial() const {
        return lossConfigGetter_().isDecomposable();
    }

    bool AutomaticHeadConfig::isSingleOutput() const {
        return lossConfigGetter_().isDecomposable();
    }

}
