/*
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */
#pragma once

#include "mlrl/seco/heuristics/heuristic.hpp"

#include <memory>

namespace seco {

    /**i
     * Allows to configure a heuristic that measures the fraction of correctly predicted labels among all labels, i.e.,
     * in contrast to the "Precision" metric, examples that are not covered by a rule are taken into account as well.
     *
     * This heuristic is used in the pruning phase of IREP (see "Incremental Reduced Error Pruning", Fürnkranz, Widmer,
     * 1994).
     */
    class AccuracyConfig : public IHeuristicConfig {
        public:

            std::unique_ptr<IHeuristicFactory> createHeuristicFactory() const override;
    };

}
