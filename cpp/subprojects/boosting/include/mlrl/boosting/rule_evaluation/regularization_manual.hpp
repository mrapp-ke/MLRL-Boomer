/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/regularization.hpp"
#include "mlrl/boosting/util/dll_exports.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a regularization term that affects the evaluation of
     * rules by manually specifying the weight of the regularization term.
     */
    class MLRLBOOSTING_API IManualRegularizationConfig {
        public:

            virtual ~IManualRegularizationConfig() {}

            /**
             * Returns the weight of the regularization term.
             *
             * @return The weight of the regularization term
             */
            virtual float32 getRegularizationWeight() const = 0;

            /**
             * Sets the weight of the regularization term.
             *
             * @param regularizationWeight  The weight of the regularization term. Must be greater than 0
             * @return                      A reference to an object of type `IManualRegularizationConfig` that allows
             *                              further configuration of the regularization term
             */
            virtual IManualRegularizationConfig& setRegularizationWeight(float32 regularizationWeight) = 0;
    };

    /**
     * Allows to configure a regularization term that affects the evaluation of rules by manually specifying the weight
     * of the regularization term.
     */
    class ManualRegularizationConfig final : public IRegularizationConfig,
                                             public IManualRegularizationConfig {
        private:

            float32 regularizationWeight_;

        public:

            ManualRegularizationConfig();

            float32 getRegularizationWeight() const override;

            IManualRegularizationConfig& setRegularizationWeight(float32 regularizationWeight) override;

            float32 getWeight() const override;
    };

}
