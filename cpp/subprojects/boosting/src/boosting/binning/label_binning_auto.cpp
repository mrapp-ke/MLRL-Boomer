#include "boosting/binning/label_binning_auto.hpp"
#include "boosting/binning/label_binning_no.hpp"


namespace boosting {

    std::unique_ptr<ILabelWiseRuleEvaluationFactory> AutomaticLabelBinningConfig::configureLabelWise() const {
        // TODO Implement
        return NoLabelBinningConfig().configureLabelWise();
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory> AutomaticLabelBinningConfig::configureExampleWise() const {
        // TODO Implement
        return NoLabelBinningConfig().configureExampleWise();
    }

}
