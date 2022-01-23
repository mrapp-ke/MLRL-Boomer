#include "boosting/binning/label_binning_auto.hpp"
#include "boosting/binning/label_binning_equal_width.hpp"
#include "boosting/binning/label_binning_no.hpp"


namespace boosting {

    AutomaticLabelBinningConfig::AutomaticLabelBinningConfig(
            const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
            const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : l1RegularizationConfigPtr_(l1RegularizationConfigPtr), l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    std::unique_ptr<ILabelWiseRuleEvaluationFactory> AutomaticLabelBinningConfig::configureLabelWise() const {
        return NoLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_).configureLabelWise();
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory> AutomaticLabelBinningConfig::configureExampleWise() const {
        return EqualWidthLabelBinningConfig(l1RegularizationConfigPtr_,l2RegularizationConfigPtr_)
            .configureExampleWise();
    }

}
