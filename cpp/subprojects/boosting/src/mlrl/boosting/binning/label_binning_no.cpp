#include "mlrl/boosting/binning/label_binning_no.hpp"

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_dynamic.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_fixed.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_complete.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_dynamic.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_fixed.hpp"

namespace boosting {

    NoLabelBinningConfig::NoLabelBinningConfig(GetterFunction<IRegularizationConfig> l1RegularizationConfigGetter,
                                               GetterFunction<IRegularizationConfig> l2RegularizationConfigGetter)
        : l1RegularizationConfigGetter_(l1RegularizationConfigGetter),
          l2RegularizationConfigGetter_(l2RegularizationConfigGetter) {}

    std::unique_ptr<IDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createDecomposableCompleteRuleEvaluationFactory() const {
        float64 l1RegularizationWeight = l1RegularizationConfigGetter_().getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigGetter_().getWeight();
        return std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight,
                                                                           l2RegularizationWeight);
    }

    std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio, uint32 minOutputs,
                                                                                uint32 maxOutputs) const {
        float64 l1RegularizationWeight = l1RegularizationConfigGetter_().getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigGetter_().getWeight();
        return std::make_unique<DecomposableFixedPartialRuleEvaluationFactory>(
          outputRatio, minOutputs, maxOutputs, l1RegularizationWeight, l2RegularizationWeight);
    }

    std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createDecomposableDynamicPartialRuleEvaluationFactory(float32 threshold,
                                                                                  float32 exponent) const {
        float64 l1RegularizationWeight = l1RegularizationConfigGetter_().getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigGetter_().getWeight();
        return std::make_unique<DecomposableDynamicPartialRuleEvaluationFactory>(
          threshold, exponent, l1RegularizationWeight, l2RegularizationWeight);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createNonDecomposableCompleteRuleEvaluationFactory(const Blas& blas,
                                                                               const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigGetter_().getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigGetter_().getWeight();
        return std::make_unique<NonDecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight,
                                                                              l2RegularizationWeight, blas, lapack);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createNonDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio,
                                                                                   uint32 minOutputs, uint32 maxOutputs,
                                                                                   const Blas& blas,
                                                                                   const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigGetter_().getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigGetter_().getWeight();
        return std::make_unique<NonDecomposableFixedPartialRuleEvaluationFactory>(
          outputRatio, minOutputs, maxOutputs, l1RegularizationWeight, l2RegularizationWeight, blas, lapack);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      NoLabelBinningConfig::createNonDecomposableDynamicPartialRuleEvaluationFactory(float32 threshold,
                                                                                     float32 exponent, const Blas& blas,
                                                                                     const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigGetter_().getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigGetter_().getWeight();
        return std::make_unique<NonDecomposableDynamicPartialRuleEvaluationFactory>(
          threshold, exponent, l1RegularizationWeight, l2RegularizationWeight, blas, lapack);
    }

}
