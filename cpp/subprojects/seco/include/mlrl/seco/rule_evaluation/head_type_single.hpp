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

            const ReadableProperty<IHeuristicConfig> heuristicConfig_;

            const ReadableProperty<IHeuristicConfig> pruningHeuristicConfig_;

        public:

            /**
             * @param heuristicConfigGetter         A `ReadableProperty` that allows to access the `IHeuristicConfig`
             *                                      that stores the configuration of the heuristic for learning rules
             * @param pruningHeuristicConfigGetter  A `ReadableProperty` that allows to access the `IHeuristicConfig`
             *                                      that stores the configuration of the heuristic for pruning rules
             */
            SingleOutputHeadConfig(ReadableProperty<IHeuristicConfig> heuristicConfigGetter,
                                   ReadableProperty<IHeuristicConfig> pruningHeuristicConfigGetter);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createStatisticsProviderFactory(
              const IRowWiseLabelMatrix& labelMatrix) const override;
    };

}
