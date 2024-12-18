/*
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */
#pragma once

#include "mlrl/seco/heuristics/heuristic.hpp"

#include <memory>

namespace seco {

    /**
     * Allows to configure a heuristic that measures the fraction of correctly predicted labels among all labels, i.e.,
     * in contrast to the "Precision" metric, examples that are not covered by a rule are taken into account as well.
     *
     * This heuristic is used in the pruning phase of IREP ("Incremental Reduced Error Pruning", Fürnkranz, Widmer 1994,
     * see https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f67ebb7b392f51076899f58c53bf57d5e71e36e9).
     */
    class AccuracyConfig : public IHeuristicConfig {
        public:

            std::unique_ptr<IHeuristicFactory> createHeuristicFactory() const override;
    };

}
