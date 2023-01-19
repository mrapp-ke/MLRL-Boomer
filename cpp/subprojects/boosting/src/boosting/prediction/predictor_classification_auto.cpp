#include "boosting/prediction/predictor_classification_auto.hpp"
#include "boosting/prediction/predictor_classification_example_wise.hpp"
#include "boosting/prediction/predictor_classification_label_wise.hpp"


namespace boosting {

    AutomaticClassificationPredictorConfig::AutomaticClassificationPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {

    }

    std::unique_ptr<IClassificationPredictorFactory> AutomaticClassificationPredictorConfig::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        if (lossConfigPtr_->isDecomposable()) {
            return LabelWiseClassificationPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createClassificationPredictorFactory(featureMatrix, numLabels);
        } else {
            return ExampleWiseClassificationPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createClassificationPredictorFactory(featureMatrix, numLabels);
        }
    }

    bool AutomaticClassificationPredictorConfig::isLabelVectorSetNeeded() const {
        if (lossConfigPtr_->isDecomposable()) {
            return LabelWiseClassificationPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .isLabelVectorSetNeeded();
        } else {
            return ExampleWiseClassificationPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .isLabelVectorSetNeeded();
        }
    }

}
