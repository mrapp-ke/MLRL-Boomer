#include "mlrl/boosting/binning/label_binning_auto.hpp"

#include "mlrl/boosting/binning/label_binning_equal_width.hpp"
#include "mlrl/boosting/binning/label_binning_no.hpp"

namespace boosting {

    AutomaticLabelBinningConfig::AutomaticLabelBinningConfig(
      const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
      const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : l1RegularizationConfigPtr_(l1RegularizationConfigPtr), l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    std::unique_ptr<IDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createDecomposableCompleteRuleEvaluationFactory() const {
        return NoLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createDecomposableCompleteRuleEvaluationFactory();
    }

    std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio,
                                                                                       uint32 minOutputs,
                                                                                       uint32 maxOutputs) const {
        return NoLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs, maxOutputs);
    }

    std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createDecomposableDynamicPartialRuleEvaluationFactory(float32 threshold,
                                                                                         float32 exponent) const {
        return NoLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createDecomposableDynamicPartialRuleEvaluationFactory(threshold, exponent);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createNonDecomposableCompleteRuleEvaluationFactory(const Blas& blas,
                                                                                      const Lapack& lapack) const {
        return EqualWidthLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createNonDecomposableCompleteRuleEvaluationFactory(blas, lapack);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createNonDecomposableFixedPartialRuleEvaluationFactory(
        float32 outputRatio, uint32 minOutputs, uint32 maxOutputs, const Blas& blas, const Lapack& lapack) const {
        return EqualWidthLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createNonDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs, maxOutputs, blas, lapack);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createNonDecomposableDynamicPartialRuleEvaluationFactory(
        float32 threshold, float32 exponent, const Blas& blas, const Lapack& lapack) const {
        return EqualWidthLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createNonDecomposableDynamicPartialRuleEvaluationFactory(threshold, exponent, blas, lapack);
    }

}
