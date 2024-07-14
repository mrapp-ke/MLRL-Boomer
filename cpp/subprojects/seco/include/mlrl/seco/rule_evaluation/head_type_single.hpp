/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/util/properties.hpp"
#include "mlrl/seco/heuristics/heuristic.hpp"
#include "mlrl/seco/rule_evaluation/head_type.hpp"

#include <memory>

namespace seco {

    /**
     * Allows to configure single-output heads that predict for a single output.
     */
    class SingleOutputHeadConfig final : public IHeadConfig {
        private:

            const GetterFunction<IHeuristicConfig> heuristicConfigGetter_;

            const GetterFunction<IHeuristicConfig> pruningHeuristicConfigGetter_;

        public:

            /**
             * @param heuristicConfigGetter         A `GetterFunction` that allows to access the `IHeuristicConfig` that
             *                                      stores the configuration of the heuristic for learning rules
             * @param pruningHeuristicConfigGetter  A `GetterFunction` that allows to access the `IHeuristicConfig` that
             *                                      stores the configuration of the heuristic for pruning rules
             */
            SingleOutputHeadConfig(GetterFunction<IHeuristicConfig> heuristicConfigGetter,
                                   GetterFunction<IHeuristicConfig> pruningHeuristicConfigGetter);

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IRowWiseLabelMatrix& labelMatrix) const override;
    };

}
