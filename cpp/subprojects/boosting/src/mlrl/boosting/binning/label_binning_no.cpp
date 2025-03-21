#include "mlrl/boosting/binning/label_binning_no.hpp"

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_dynamic.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_fixed.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_complete.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_dynamic.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_fixed.hpp"

namespace boosting {

    NoLabelBinningConfig::NoLabelBinningConfig(ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                                               ReadableProperty<IRegularizationConfig> l2RegularizationConfig)
        : l1RegularizationConfig_(l1RegularizationConfig), l2RegularizationConfig_(l2RegularizationConfig) {}

    std::unique_ptr<IDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createDecomposableCompleteRuleEvaluationFactory() const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        return std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight,
                                                                           l2RegularizationWeight);
    }

    std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio, uint32 minOutputs,
                                                                                uint32 maxOutputs) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        return std::make_unique<DecomposableFixedPartialRuleEvaluationFactory>(
          outputRatio, minOutputs, maxOutputs, l1RegularizationWeight, l2RegularizationWeight);
    }

    std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createDecomposableDynamicPartialRuleEvaluationFactory(float32 threshold,
                                                                                  float32 exponent) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        return std::make_unique<DecomposableDynamicPartialRuleEvaluationFactory>(
          threshold, exponent, l1RegularizationWeight, l2RegularizationWeight);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createNonDecomposableCompleteRuleEvaluationFactory(
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        return std::make_unique<NonDecomposableCompleteRuleEvaluationFactory>(
          l1RegularizationWeight, l2RegularizationWeight, blasFactory, lapackFactory);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createNonDecomposableFixedPartialRuleEvaluationFactory(
        float32 outputRatio, uint32 minOutputs, uint32 maxOutputs, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        return std::make_unique<NonDecomposableFixedPartialRuleEvaluationFactory>(
          outputRatio, minOutputs, maxOutputs, l1RegularizationWeight, l2RegularizationWeight, blasFactory,
          lapackFactory);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createNonDecomposableDynamicPartialRuleEvaluationFactory(
        float32 threshold, float32 exponent, const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        return std::make_unique<NonDecomposableDynamicPartialRuleEvaluationFactory>(
          threshold, exponent, l1RegularizationWeight, l2RegularizationWeight, blasFactory, lapackFactory);
    }

}
