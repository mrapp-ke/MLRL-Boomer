/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Defines an interface for all classes that allow to configure a heuristic that measures the fraction of
     * incorrectly predicted labels among all labels that are covered by a rule.
     *
     * This heuristic is equivalent to the pruning heuristic used by RIPPER ("Fast Effective Rule Induction", Cohen
     * 1995, see https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.294.7522&rep=rep1&type=pdf). A proof is
     * provided in the paper "Roc 'n' Rule Learning — Towards a Better Understanding of Covering Algorithms", Fürnkranz,
     * Flach 2005 (see https://link.springer.com/content/pdf/10.1007/s10994-005-5011-x.pdf).
     */
    class IPrecisionConfig {

        public:

            virtual ~IPrecisionConfig() { };

    };

    /**
     * Allows to configure a heuristic that measures the fraction of incorrectly predicted labels among all labels that
     * are covered by a rule.
     */
    class PrecisionConfig final : public IHeuristicConfig, public IPrecisionConfig {

        public:

            std::unique_ptr<IHeuristicFactory> create() const override;

    };

}
