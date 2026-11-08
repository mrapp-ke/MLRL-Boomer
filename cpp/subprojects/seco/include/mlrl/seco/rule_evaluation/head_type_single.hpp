/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/simd.hpp"
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

            const ReadableProperty<ISimdConfig> simdConfig_;

        public:

            /**
             * @param heuristicConfig           A `ReadableProperty` that allows to access the `IHeuristicConfig` that
             *                                  stores the configuration of the heuristic for learning rules
             * @param pruningHeuristicConfig    A `ReadableProperty` that allows to access the `IHeuristicConfig` that
             *                                  stores the configuration of the heuristic for pruning rules
             * @param simdConfig                A `ReadableProperty` that allows to access the `ISimdConfig` that stores
             *                                  the configuration of SIMD operations
             */
            SingleOutputHeadConfig(ReadableProperty<IHeuristicConfig> heuristicConfig,
                                   ReadableProperty<IHeuristicConfig> pruningHeuristicConfig,
                                   ReadableProperty<ISimdConfig> simdConfig);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createStatisticsProviderFactory(
              const IRowWiseLabelMatrix& labelMatrix) const override;
    };

}
