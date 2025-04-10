/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/lift_functions/lift_function.hpp"
#include "mlrl/seco/util/dll_exports.hpp"

#include <memory>

namespace seco {

    /**
     * Defines an interface for all classes that allow to configure a lift function that monotonously increases
     * according to the natural logarithm of the number of labels for which a rule predicts.
     */
    class MLRLSECO_API IKlnLiftFunctionConfig {
        public:

            virtual ~IKlnLiftFunctionConfig() {}

            /**
             * Returns the value of the parameter "k", which affects the steepness of the lift function.
             *
             * @return The value of the parameter "k"
             */
            virtual float32 getK() const = 0;

            /**
             * Sets the value of the parameter "k", which affects the steepness of the lift function.
             *
             * @param k The value of the parameter "k". The steepness of the lift function increases with larger values
             *          for "k". Must be greater than 0
             * @return  A reference to an object of type `IKlnLiftFunctionConfig` that allows further configuration of
             *          the lift function
             */
            virtual IKlnLiftFunctionConfig& setK(float32 k) = 0;
    };

    /**
     * Allows to configure a lift function that monotonously increases according to the natural logarithm of the number
     * of labels for which a rule predicts.
     */
    class KlnLiftFunctionConfig final : public ILiftFunctionConfig,
                                        public IKlnLiftFunctionConfig {
        private:

            float32 k_;

        public:

            KlnLiftFunctionConfig();

            float32 getK() const override;

            IKlnLiftFunctionConfig& setK(float32 k) override;

            std::unique_ptr<ILiftFunctionFactory> createLiftFunctionFactory(
              const IRowWiseLabelMatrix& labelMatrix) const override;
    };

}
