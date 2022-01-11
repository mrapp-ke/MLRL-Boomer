/*
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Allows to configure a heuristic that measures the fraction of incorrectly predicted labels among all labels,
     * i.e., in contrast to the precision metric, examples that are not covered by a rule are taken into account as
     * well.
     *
     * This heuristic is used in the pruning phase of IREP ("Incremental Reduced Error Pruning", Fürnkranz, Widmer 1994,
     * see https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.7813&rep=rep1&type=pdf).
     */
    class AccuracyConfig final : public IHeuristicConfig {

    };

    /**
     * Allows to create instances of the type `IHeuristic` that measure the fraction of incorrectly predicted labels
     * among all labels, i.e., in contrast to the precision metric, examples that are not covered by a rule are taken
     * into account as well.
     */
    class AccuracyFactory final : public IHeuristicFactory {

        public:

            std::unique_ptr<IHeuristic> create() const override;

    };

}
