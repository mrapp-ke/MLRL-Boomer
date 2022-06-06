/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_induction/rule_induction.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "common/macros.hpp"


/**
 * Defines an interface for all classes that allow to configure an algorithm for the induction of individual rules that
 * uses a top-down greedy search.
 */
class MLRLCOMMON_API IGreedyTopDownRuleInductionConfig {

    public:

        virtual ~IGreedyTopDownRuleInductionConfig() { };

        /**
         * Returns the minimum number of training examples that must be covered by a rule.
         *
         * @return The minimum number of training examples that must be covered by a rule.
         */
        virtual uint32 getMinCoverage() const = 0;

        /**
         * Sets the minimum number of training examples that must be covered by a rule.
         *
         * @param minCoverage   The minimum number of training examples that must be covered by a rule. Must be at least
         *                      1
         * @return              A reference to an object of type `IGreedyTopDownRuleInductionConfig` that allows further
         *                      configuration of the algorithm for the induction of individual rules
         */
        virtual IGreedyTopDownRuleInductionConfig& setMinCoverage(uint32 minCoverage) = 0;

        /**
         * Returns the maximum number of conditions to be included in a rule's body.
         *
         * @return The maximum number of conditions to be included in a rule's body or 0, if the number of conditions is
         *         not restricted
         */
        virtual uint32 getMaxConditions() const = 0;

        /**
         * Sets the maximum number of conditions to be included in a rule's body.
         *
         * @param maxConditions The maximum number of conditions to be included in a rule's body. Must be at least 1 or
         *                      0, if the number of conditions should not be restricted
         * @return              A reference to an object of type `IGreedyTopDownRuleInductionConfig` that allows further
         *                      configuration of the algorithm for the induction of individual rules
         */
        virtual IGreedyTopDownRuleInductionConfig& setMaxConditions(uint32 maxConditions) = 0;

        /**
         * Returns the maximum number of times, the head of a rule may be refinement after a new condition has been
         * added to its body.
         *
         * @return The maximum number of times, the head of a rule may be refined or 0, if the number of refinements is
         *         not restricted
         */
        virtual uint32 getMaxHeadRefinements() const = 0;

        /**
         * Sets the maximum number of times, the head of a rule may be refined after a new condition has been added to
         * its body.
         *
         * @param maxHeadRefinements    The maximum number of times, the head of a rule may be refined. Must be at least
         *                              1 or 0, if the number of refinements should not be restricted
         * @return                      A reference to an object of type `IGreedyTopDownRuleInductionConfig` that allows
         *                              further configuration of the algorithm for the induction of individual rules
         */
        virtual IGreedyTopDownRuleInductionConfig& setMaxHeadRefinements(uint32 maxHeadRefinements) = 0;

        /**
         * Returns whether the predictions of rules are recalculated on all training examples, if some of the examples
         * have zero weights, or not.
         *
         * @return True, if the predictions of rules are recalculated on all training examples, false otherwise
         */
        virtual bool arePredictionsRecalculated() const = 0;

        /**
         * Sets whether the predictions of rules should be recalculated on all training examples, if some of the
         * examples have zero weights, or not.
         *
         * @param recalculatePredictions    True, if the predictions of rules should be recalculated on all training
         *                                  examples, false otherwise
         * @return                          A reference to an object of type `IGreedyTopDownRuleInductionConfig` that
         *                                  allows further configuration of the algorithm for the induction of
         *                                  individual rules
         */
        virtual IGreedyTopDownRuleInductionConfig& setRecalculatePredictions(bool recalculatePredictions) = 0;

};

/**
 * Allows to configure an algorithm for the induction of individual rules that uses a top-down greedy search.
 */
class GreedyTopDownRuleInductionConfig final : public IRuleInductionConfig, public IGreedyTopDownRuleInductionConfig {

    private:

        uint32 minCoverage_;

        uint32 maxConditions_;

        uint32 maxHeadRefinements_;

        bool recalculatePredictions_;

        const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

    public:

        /**
         * @param multiThreadingConfigPtr A reference to an unique pointer that stores the configuration of the
         *                                multi-threading behavior that should be used for the parallel refinement of
         *                                rules
         */
        GreedyTopDownRuleInductionConfig(const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

        uint32 getMinCoverage() const override;

        IGreedyTopDownRuleInductionConfig& setMinCoverage(uint32 minCoverage) override;

        uint32 getMaxConditions() const override;

        IGreedyTopDownRuleInductionConfig& setMaxConditions(uint32 maxConditions) override;

        uint32 getMaxHeadRefinements() const override;

        IGreedyTopDownRuleInductionConfig& setMaxHeadRefinements(uint32 maxHeadRefinements) override;

        bool arePredictionsRecalculated() const override;

        IGreedyTopDownRuleInductionConfig& setRecalculatePredictions(bool recalculatePredictions) override;

        std::unique_ptr<IRuleInductionFactory> createRuleInductionFactory(
            const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const override;

};
