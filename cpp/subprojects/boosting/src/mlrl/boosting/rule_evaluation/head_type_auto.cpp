#include "mlrl/boosting/rule_evaluation/head_type_auto.hpp"

#include "mlrl/boosting/rule_evaluation/head_type_complete.hpp"
#include "mlrl/boosting/rule_evaluation/head_type_single.hpp"

#include <functional>

namespace boosting {

    template<typename StatisticType>
    class Preset final : public IHeadConfig::IPreset<StatisticType> {
        private:

            typedef std::function<std::unique_ptr<IHeadConfig::IPreset<StatisticType>>(const IHeadConfig&)>
              PresetFunction;

            ReadableProperty<ILabelBinningConfig> labelBinningConfig_;

            ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

            ReadableProperty<IRegularizationConfig> l1RegularizationConfig_;

            ReadableProperty<IRegularizationConfig> l2RegularizationConfig_;

            PresetFunction presetFunction_;

        public:

            Preset(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                   ReadableProperty<IMultiThreadingConfig> multiThreadingConfig,
                   ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                   ReadableProperty<IRegularizationConfig> l2RegularizationConfig, PresetFunction presetFunction)
                : labelBinningConfig_(labelBinningConfig), multiThreadingConfig_(multiThreadingConfig),
                  l1RegularizationConfig_(l1RegularizationConfig), l2RegularizationConfig_(l2RegularizationConfig),
                  presetFunction_(presetFunction) {}

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<IDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr)
              const override {
                CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                              l2RegularizationConfig_);
                return presetFunction_(headConfig)
                  ->createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, lossFactoryPtr,
                                                                  evaluationMeasureFactoryPtr);
            }

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<ISparseDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<ISparseEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr)
              const override {
                if (labelMatrix.getNumOutputs() > 1) {
                    SingleOutputHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_,
                                                      l1RegularizationConfig_, l2RegularizationConfig_);
                    return presetFunction_(headConfig)
                      ->createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, lossFactoryPtr,
                                                                      evaluationMeasureFactoryPtr);
                } else {
                    CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                                  l2RegularizationConfig_);
                    return presetFunction_(headConfig)
                      ->createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, lossFactoryPtr,
                                                                      evaluationMeasureFactoryPtr);
                }
            }

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<INonDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const override {
                CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                              l2RegularizationConfig_);
                return presetFunction_(headConfig)
                  ->createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, lossFactoryPtr,
                                                                  evaluationMeasureFactoryPtr, blasFactory,
                                                                  lapackFactory);
            }

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              std::unique_ptr<IDecomposableRegressionLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr)
              const override {
                CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                              l2RegularizationConfig_);
                return presetFunction_(headConfig)
                  ->createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix, lossFactoryPtr,
                                                              evaluationMeasureFactoryPtr);
            }

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              std::unique_ptr<INonDecomposableRegressionLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const override {
                CompleteHeadConfig headConfig(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                              l2RegularizationConfig_);
                return presetFunction_(headConfig)
                  ->createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix, lossFactoryPtr,
                                                              evaluationMeasureFactoryPtr, blasFactory, lapackFactory);
            }
    };

    AutomaticHeadConfig::AutomaticHeadConfig(ReadableProperty<ILossConfig> lossConfig,
                                             ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                             ReadableProperty<IMultiThreadingConfig> multiThreadingConfig,
                                             ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                                             ReadableProperty<IRegularizationConfig> l2RegularizationConfig)
        : lossConfig_(lossConfig), labelBinningConfig_(labelBinningConfig), multiThreadingConfig_(multiThreadingConfig),
          l1RegularizationConfig_(l1RegularizationConfig), l2RegularizationConfig_(l2RegularizationConfig) {}

    std::unique_ptr<IHeadConfig::IPreset<float32>> AutomaticHeadConfig::create32Bit() const {
        return std::make_unique<Preset<float32>>(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                                 l2RegularizationConfig_, [](const IHeadConfig& config) {
            return config.create32Bit();
        });
    }

    std::unique_ptr<IHeadConfig::IPreset<float64>> AutomaticHeadConfig::create64Bit() const {
        return std::make_unique<Preset<float64>>(labelBinningConfig_, multiThreadingConfig_, l1RegularizationConfig_,
                                                 l2RegularizationConfig_, [](const IHeadConfig& config) {
            return config.create64Bit();
        });
    }

    bool AutomaticHeadConfig::isPartial() const {
        return lossConfig_.get().isDecomposable() && lossConfig_.get().isSparse();
    }

    bool AutomaticHeadConfig::isSingleOutput() const {
        return lossConfig_.get().isDecomposable() && lossConfig_.get().isSparse();
    }

}
