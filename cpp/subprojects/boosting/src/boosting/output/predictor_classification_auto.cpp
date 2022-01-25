#include "boosting/output/predictor_classification_auto.hpp"
#include "boosting/output/predictor_classification_example_wise.hpp"
#include "boosting/output/predictor_classification_label_wise.hpp"
#include "boosting/losses/loss_example_wise.hpp"


namespace boosting {

    static inline constexpr bool isExampleWisePredictorPreferred(const ILossConfig* lossConfig) {
        return dynamic_cast<const IExampleWiseLossConfig*>(lossConfig) != nullptr;
    }

    AutomaticClassificationPredictorConfig::AutomaticClassificationPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : lossConfigPtr_(lossConfigPtr) {

    }

    std::unique_ptr<IClassificationPredictorFactory> AutomaticClassificationPredictorConfig::createClassificationPredictorFactory() const {
        if (isExampleWisePredictorPreferred(lossConfigPtr_.get())) {
            return ExampleWiseClassificationPredictorConfig(lossConfigPtr_).createClassificationPredictorFactory();
        } else {
            return LabelWiseClassificationPredictorConfig(lossConfigPtr_).createClassificationPredictorFactory();
        }
    }

    std::unique_ptr<ILabelSpaceInfo> AutomaticClassificationPredictorConfig::createLabelSpaceInfo(
            const IRowWiseLabelMatrix& labelMatrix) const {
        if (isExampleWisePredictorPreferred(lossConfigPtr_.get())) {
            return ExampleWiseClassificationPredictorConfig(lossConfigPtr_).createLabelSpaceInfo(labelMatrix);
        } else {
            return LabelWiseClassificationPredictorConfig(lossConfigPtr_).createLabelSpaceInfo(labelMatrix);
        }
    }

}
