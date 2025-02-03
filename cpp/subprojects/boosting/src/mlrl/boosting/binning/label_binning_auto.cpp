#include "mlrl/boosting/binning/label_binning_auto.hpp"

#include "mlrl/boosting/binning/label_binning_equal_width.hpp"
#include "mlrl/boosting/binning/label_binning_no.hpp"

namespace boosting {

    AutomaticLabelBinningConfig::AutomaticLabelBinningConfig(
      ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
      ReadableProperty<IRegularizationConfig> l2RegularizationConfig)
        : l1RegularizationConfig_(l1RegularizationConfig), l2RegularizationConfig_(l2RegularizationConfig) {}

    std::unique_ptr<IDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createDecomposableCompleteRuleEvaluationFactory() const {
        return NoLabelBinningConfig(l1RegularizationConfig_, l2RegularizationConfig_)
          .createDecomposableCompleteRuleEvaluationFactory();
    }

    std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio,
                                                                                       uint32 minOutputs,
                                                                                       uint32 maxOutputs) const {
        return NoLabelBinningConfig(l1RegularizationConfig_, l2RegularizationConfig_)
          .createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs, maxOutputs);
    }

    std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createDecomposableDynamicPartialRuleEvaluationFactory(float32 threshold,
                                                                                         float32 exponent) const {
        return NoLabelBinningConfig(l1RegularizationConfig_, l2RegularizationConfig_)
          .createDecomposableDynamicPartialRuleEvaluationFactory(threshold, exponent);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createNonDecomposableCompleteRuleEvaluationFactory(
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        return EqualWidthLabelBinningConfig(l1RegularizationConfig_, l2RegularizationConfig_)
          .createNonDecomposableCompleteRuleEvaluationFactory(blasFactory, lapackFactory);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createNonDecomposableFixedPartialRuleEvaluationFactory(
        float32 outputRatio, uint32 minOutputs, uint32 maxOutputs, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory) const {
        return EqualWidthLabelBinningConfig(l1RegularizationConfig_, l2RegularizationConfig_)
          .createNonDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs, maxOutputs, blasFactory,
                                                                  lapackFactory);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createNonDecomposableDynamicPartialRuleEvaluationFactory(
        float32 threshold, float32 exponent, const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        return EqualWidthLabelBinningConfig(l1RegularizationConfig_, l2RegularizationConfig_)
          .createNonDecomposableDynamicPartialRuleEvaluationFactory(threshold, exponent, blasFactory, lapackFactory);
    }

}
