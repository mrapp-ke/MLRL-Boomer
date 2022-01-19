/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Defines an interface for all classes that allow to configure a heuristic that measures the fraction of uncovered
     * labels among all labels for which a rule's prediction is (or would be) correct, i.e., for which the ground truth
     * is equal to the rule's prediction.
     */
    class IRecallConfig {

        public:

            virtual ~IRecallConfig() { };

    };

    /**
     * Allows to configure a heuristic that measures the fraction of uncovered labels among all labels for which a
     * rule's prediction is (or would be) correct, i.e., for which the ground truth is equal to the rule's prediction.
     */
    class RecallConfig final : public IHeuristicConfig, public IRecallConfig {

        public:

            std::unique_ptr<IHeuristicFactory> create() const override;

    };

}
