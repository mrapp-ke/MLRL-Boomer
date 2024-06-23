/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

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

            const std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr_;

            const std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr_;

            const std::unique_ptr<ILiftFunctionConfig>& liftFunctionConfigPtr_;

        public:

            /**
             * @param heuristicConfigPtr        A reference to an unique pointer that stores the configuration of the
             *                                  heuristic for learning rules
             * @param pruningHeuristicConfigPtr A reference to an unique pointer that stores the configuration of the
             *                                  heuristic for pruning rules
             * @param liftFunctionConfigPtr     A reference to an unique pointer that stores the configuration of the
             *                                  lift function that should affect the quality of rules, depending on the
             *                                  number of labels for which they predict
             */
            PartialHeadConfig(const std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr,
                              const std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr,
                              const std::unique_ptr<ILiftFunctionConfig>& liftFunctionConfigPtr);

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IRowWiseLabelMatrix& labelMatrix) const override;
    };

}
