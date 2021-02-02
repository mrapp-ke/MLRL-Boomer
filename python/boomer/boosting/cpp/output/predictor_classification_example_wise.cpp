#include "predictor_classification_example_wise.h"
#include <iostream> // TODO Remove


namespace boosting {

    void ExampleWiseClassificationPredictor::addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr) {
        labelVectors_.emplace(std::move(labelVectorPtr));
    }

    void ExampleWiseClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                     CContiguousView<uint8>& predictionMatrix,
                                                     const RuleModel& model) const {
        // TODO
    }

    void ExampleWiseClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                     CContiguousView<uint8>& predictionMatrix,
                                                     const RuleModel& model) const {
        // TODO
    }

}
