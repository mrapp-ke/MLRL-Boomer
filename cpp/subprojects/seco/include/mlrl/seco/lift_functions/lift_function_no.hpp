/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/dll_exports.hpp"
#include "mlrl/seco/lift_functions/lift_function.hpp"

namespace seco {

    /**
     * Allows to configure a lift function that does not affect the quality of rules.
     */
    class NoLiftFunctionConfig final : public ILiftFunctionConfig {
        public:

            std::unique_ptr<ILiftFunctionFactory> createLiftFunctionFactory(
              const IRowWiseLabelMatrix& labelMatrix) const override;
    };

}
