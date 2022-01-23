/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_classification.hpp"


namespace boosting {

    /**
     * Allows to configure a predictor that automatically decides for a method that is used to predict whether
     * individual labels of given query examples are relevant or not
     */
    class AutomaticClassificationPredictorConfig : public IClassificationPredictorConfig {

        public:

            std::unique_ptr<IClassificationPredictorFactory> configure() const override;

    };

}
