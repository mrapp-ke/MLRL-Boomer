#include "boosting/binning/label_binning_no.hpp"
#include "boosting/rule_evaluation/rule_evaluation_example_wise_complete.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_complete.hpp"


namespace boosting {

    std::unique_ptr<ILabelWiseRuleEvaluationFactory> NoLabelBinningConfig::configureLabelWise() const {
        uint32 l1RegularizationWeight = 0;  // TODO Use correct value
        uint32 l2RegularizationWeight = 0;  // TODO Use correct value
        return std::make_unique<LabelWiseCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory> NoLabelBinningConfig::configureExampleWise() const {
        uint32 l1RegularizationWeight = 0;  // TODO Use correct value
        uint32 l2RegularizationWeight = 0;  // TODO Use correct value
        std::unique_ptr<Blas> blasPtr = nullptr;  // TODO
        std::unique_ptr<Lapack> lapackPtr = nullptr;  // TODO
        return std::make_unique<ExampleWiseCompleteRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight, std::move(blasPtr),
                                                                          std::move(lapackPtr));
    }

}
