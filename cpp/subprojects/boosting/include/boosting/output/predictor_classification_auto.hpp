/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_classification.hpp"
#include "boosting/losses/loss.hpp"


namespace boosting {

    /**
     * Allows to configure a predictor that automatically decides for a method that is used to predict whether
     * individual labels of given query examples are relevant or not
     */
    class AutomaticClassificationPredictorConfig : public IClassificationPredictorConfig {

        private:

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

        public:

            /**
             * @param lossConfigPtr A reference to an unique pointer that stores the configuration of the loss function
             */
            AutomaticClassificationPredictorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr);

            std::unique_ptr<IClassificationPredictorFactory> configure() const override;

    };

}
