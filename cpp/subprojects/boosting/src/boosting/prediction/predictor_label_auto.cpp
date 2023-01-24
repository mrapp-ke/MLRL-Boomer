#include "boosting/prediction/predictor_label_auto.hpp"
#include "boosting/prediction/predictor_label_example_wise.hpp"
#include "boosting/prediction/predictor_label_label_wise.hpp"


namespace boosting {

    AutomaticLabelPredictorConfig::AutomaticLabelPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {

    }

    std::unique_ptr<ILabelPredictorFactory> AutomaticLabelPredictorConfig::createPredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        if (lossConfigPtr_->isDecomposable()) {
            return LabelWiseLabelPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createPredictorFactory(featureMatrix, numLabels);
        } else {
            return ExampleWiseLabelPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createPredictorFactory(featureMatrix, numLabels);
        }
    }

    std::unique_ptr<ISparseLabelPredictorFactory> AutomaticLabelPredictorConfig::createSparsePredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        if (lossConfigPtr_->isDecomposable()) {
            return LabelWiseLabelPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createSparsePredictorFactory(featureMatrix, numLabels);
        } else {
            return ExampleWiseLabelPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createSparsePredictorFactory(featureMatrix, numLabels);
        }
    }

    bool AutomaticLabelPredictorConfig::isLabelVectorSetNeeded() const {
        if (lossConfigPtr_->isDecomposable()) {
            return LabelWiseLabelPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .isLabelVectorSetNeeded();
        } else {
            return ExampleWiseLabelPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .isLabelVectorSetNeeded();
        }
    }

}
