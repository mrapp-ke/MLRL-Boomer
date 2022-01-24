#include "boosting/output/predictor_classification_auto.hpp"
#include "boosting/output/predictor_classification_example_wise.hpp"
#include "boosting/output/predictor_classification_label_wise.hpp"
#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    AutomaticClassificationPredictorConfig::AutomaticClassificationPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : lossConfigPtr_(lossConfigPtr) {

    }

    std::unique_ptr<IClassificationPredictorFactory> AutomaticClassificationPredictorConfig::configure() const {
        if (dynamic_cast<const ILabelWiseLossConfig*>(lossConfigPtr_.get())) {
            return LabelWiseClassificationPredictorConfig().configure();
        } else {
            return ExampleWiseClassificationPredictorConfig().configure();
        }
    }

}
