/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/heuristics/heuristic.hpp"

#include <memory>

namespace seco {

    /**
     * Allows to configure a heuristic that measures the fraction of correctly predicted labels among all labels that
     * are covered by a rule.
     *
     * This heuristic is equivalent to RIPPER's pruning heuristic (see "Fast Effective Rule Induction", Cohen, 1995). A
     * proof is provided in the paper "Roc 'n' Rule Learning — Towards a Better Understanding of Covering Algorithms",
     * Fürnkranz, Flach, 2005.
     */
    class PrecisionConfig final : public IHeuristicConfig {
        public:

            std::unique_ptr<IHeuristicFactory> createHeuristicFactory() const override;
    };

}
