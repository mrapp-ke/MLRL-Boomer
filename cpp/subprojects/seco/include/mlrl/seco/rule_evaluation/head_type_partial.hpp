/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/util/properties.hpp"
#include "mlrl/seco/heuristics/heuristic.hpp"
#include "mlrl/seco/lift_functions/lift_function.hpp"
#include "mlrl/seco/rule_evaluation/head_type.hpp"

#include <memory>

namespace seco {

    /**
     * Allows to configure partial rule heads that predict for a subset of the available labels.
     */
    class PartialHeadConfig final : public IHeadConfig {
        private:

            const ReadableProperty<IHeuristicConfig> heuristicConfig_;

            const ReadableProperty<IHeuristicConfig> pruningHeuristicConfig_;

            const ReadableProperty<ILiftFunctionConfig> liftFunctionConfig_;

        public:

            /**
             * @param heuristicConfigGetter         A `ReadableProperty` that allows to access the `IHeuristicConfig`
             *                                      that stores the configuration of the heuristic for learning rules
             * @param pruningHeuristicConfigGetter  A `ReadableProperty` that allows to access the `IHeuristicConfig`
             *                                      that stores the configuration of the heuristic for pruning rules
             * @param liftFunctionConfigGetter      A `ReadableProperty` that allows to access the `ILiftFunctionConfig`
             *                                      that stores the configuration of the lift function that should
             *                                      affect the quality of rules, depending on the number of labels for
             *                                      which they predict
             */
            PartialHeadConfig(ReadableProperty<IHeuristicConfig> heuristicConfigGetter,
                              ReadableProperty<IHeuristicConfig> pruningHeuristicConfigGetter,
                              ReadableProperty<ILiftFunctionConfig> liftFunctionConfigGetter);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createStatisticsProviderFactory(
              const IRowWiseLabelMatrix& labelMatrix) const override;
    };

}
