/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_induction/rule_induction.hpp"


/**
 * Allows to configure an algorithm for the induction of individual rules that uses a top-down greedy search.
 */
class TopDownRuleInductionConfig : public IRuleInductionConfig {

    private:

        uint32 minCoverage_;

        uint32 maxConditions_;

        uint32 maxHeadRefinements_;

        bool recalculatePredictions_;

        uint32 numThreads_;

    public:

        TopDownRuleInductionConfig();

        /**
         * Returns the minimum number of training examples that must be covered by a rule.
         *
         * @return The minimum number of training examples that must be covered by a rule.
         */
        uint32 getMinCoverage() const;

        /**
         * Sets the minimum number of training examples that must be covered by a rule.
         *
         * @param minCoverage   The minimum number of training examples that must be covered by a rule. Must be at least
         *                      1
         * @return              A reference to an object of type `TopDownRuleInduction` that allows further
         *                      configuration of the algorithm for the induction of individual rules
         */
        TopDownRuleInductionConfig& setMinCoverage(uint32 minCoverage);

        /**
         * Returns the maximum number of conditions to be included in a rule's body.
         *
         * @return The maximum number of conditions to be included in a rule's body or 0, if the number of conditions is
         *         not restricted
         */
        uint32 getMaxConditions() const;

        /**
         * Sets the maximum number of conditions to be included in a rule's body.
         *
         * @param maxConditions The maximum number of conditions to be included in a rule's body. Must be at least 1 or
         *                      0, if the number of conditions should not be restricted
         * @return              A reference to an object of type `TopDownRuleInduction` that allows further
         *                      configuration of the algorithm for the induction of individual rules
         */
        TopDownRuleInductionConfig& setMaxConditions(uint32 maxConditions);

        /**
         * Returns the maximum number of times, the head of a rule may be refinement after a new condition has been
         * added to its body.
         *
         * @return The maximum number of times, the head of a rule may be refined or 0, if the number of refinements is
         *         not restricted
         */
        uint32 getMaxHeadRefinements() const;

        /**
         * Sets the maximum number of times, the head of a rule may be refined after a new condition has been added to
         * its body.
         *
         * @param maxHeadRefinement The maximum number of times, the head of a rule may be refined. Must be at least 1
         *                          or 0, if the number of refinements should not be restricted
         * @return                  A reference to an object of type `TopDownRuleInduction` that allows further
         *                          configuration of the algorithm for the induction of individual rules
         */
        TopDownRuleInductionConfig& setMaxHeadRefinements(uint32 maxHeadRefinements);

        /**
         * Returns whether the predictions of rules are recalculated on all training examples, if some of the examples
         * have zero weights, or not.
         *
         * @return True, if the predictions of rules are recalculated on all training examples, false otherwise
         */
        bool getRecalculatePredictions() const;

        /**
         * Sets whether the predictions of rules should be recalculated on all training examples, if some of the
         * examples have zero weights, or not.
         *
         * @param recalculatePredictions    True, if the predictions of rules should be recalculated on all training
         *                                  examples, false otherwise
         * @return                          A reference to an object of type `TopDownRuleInduction` that allows further
         *                                  configuration of the algorithm for the induction of individual rules
         */
        TopDownRuleInductionConfig& setRecalculatePredictions(bool recalculatePredictions);

        /**
         * Returns the number of CPU threads to be used to search for potential refinements of rules in parallel.
         *
         * @return The number of CPU threads to be used
         */
        uint32 getNumThreads() const;

        /**
         * Sets the number of CPU threads to be used to search for potential refinements of rules in parallel.
         *
         * @param numThreads    The number of CPU threads to be used. Must be at least 1
         * @return              A reference to an object of type `TopDownRuleInduction` that allows further
         *                      configuration of the algorithm for the induction of individual rules
         */
        TopDownRuleInductionConfig& setNumThreads(uint32 numThreads);

};

/**
 * Allows to create instances of the type `IRuleInduction` that induce classification rules by using a top-down greedy
 * search, where new conditions are added iteratively to the (initially empty) body of a rule. At each iteration, the
 * refinement that improves the rule the most is chosen. The search stops if no refinement results in an improvement.
 */
class TopDownRuleInductionFactory final : public IRuleInductionFactory {

    private:

        uint32 minCoverage_;

        uint32 maxConditions_;

        uint32 maxHeadRefinements_;

        bool recalculatePredictions_;

        uint32 numThreads_;

    public:

        /**
         * @param minCoverage               The minimum number of training examples that must be covered by a rule. Must
         *                                  be at least 1
         * @param maxConditions             The maximum number of conditions to be included in a rule's body. Must be at
         *                                  least 1 or 0, if the number of conditions should not be restricted
         * @param maxHeadRefinements        The maximum number of times, the head of a rule may be refined after a new
         *                                  condition has been added to its body. Must be at least 1 or 0, if the number
         *                                  of refinements should not be restricted
         * @param recalculatePredictions    True, if the predictions of rules should be recalculated on all training
         *                                  examples, if some of the examples have zero weights, false otherwise
         * @param numThreads                The number of CPU threads to be used to search for potential refinements of
         *                                  a rule in parallel. Must be at least 1
         */
        // TODO Check if it is better to pass a config here and store it as a member variable!? This probably allows to get rid of the config's getter methods when defined as a friend class
        TopDownRuleInductionFactory(uint32 minCoverage, uint32 maxConditions, uint32 maxHeadRefinements,
                                    bool recalculatePredictions, uint32 numThreads);

        std::unique_ptr<IRuleInduction> create() const override;

};
