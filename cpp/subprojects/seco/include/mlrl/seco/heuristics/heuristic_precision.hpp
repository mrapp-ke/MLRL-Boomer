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
     * This heuristic is equivalent to RIPPER's pruning heuristic ("Fast Effective Rule Induction", Cohen 1995, see
     * https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=6c34b7b0441bff66cce2418d36acfd9776ad7bd2). A proof
     * is provided in the paper "Roc 'n' Rule Learning — Towards a Better Understanding of Covering Algorithms",
     * Fürnkranz, Flach 2005 (see https://link.springer.com/content/pdf/10.1007/s10994-005-5011-x.pdf).
     */
    class PrecisionConfig final : public IHeuristicConfig {
        public:

            std::unique_ptr<IHeuristicFactory> createHeuristicFactory() const override;
    };

}
