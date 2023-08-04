/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/heuristics/heuristic.hpp"
#include "mlrl/seco/rule_evaluation/head_type.hpp"

namespace seco {

    /**
     * Allows to configure single-label rule heads that predict for a single label.
     */
    class SingleLabelHeadConfig final : public IHeadConfig {
        private:

            const std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr_;

            const std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr_;

        public:

            /**
             * @param heuristicConfigPtr        A reference to an unique pointer that stores the configuration of the
             *                                  heuristic for learning rules
             * @param pruningHeuristicConfigPtr A reference to an unique pointer that stores the configuration of the
             *                                  heuristic for pruning rules
             */
            SingleLabelHeadConfig(const std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr,
                                  const std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr);

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IRowWiseLabelMatrix& labelMatrix) const override;
    };

}
