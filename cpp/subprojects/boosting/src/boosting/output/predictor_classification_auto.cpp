#include "boosting/output/predictor_classification_auto.hpp"
#include "boosting/output/predictor_classification_label_wise.hpp"


namespace boosting {

    std::unique_ptr<IClassificationPredictorFactory> AutomaticClassificationPredictorConfig::configure() const {
        // TODO Implement
        return LabelWiseClassificationPredictorConfig().configure();
    }

}
