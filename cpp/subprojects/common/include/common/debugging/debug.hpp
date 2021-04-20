/**
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */

#pragma once

#include <common/model/condition_list.hpp>
#include "common/thresholds/thresholds_subset.hpp"



/**
 * Sets the flag for all debugging prints.
 */
void setFullFlag();

/**
 * Sets the flag if the coverage mask should be included in the debugging prints.
 */
void setCMFlag();

/**
 * Sets the flag if the sets distribution should be included in the debugging prints.
 */
void setDistFlag();

/**
 * Sets the flag if the head scores should be included in the debugging prints.
 */
void setHSFlag();

/**
 * Sets the flag if the covered labels should be included in the debugging prints.
 */
void setLCFlag();

/**
 * Sets the flag if the alternative rules should be included in the debugging prints.
 */
void setRIFlag();

/**
 * Sets the flag if the confusion matrices should be included in the debugging prints.
 */
void setConfusionFlag();

/**
 * Sets the flag if the output used to check the pruning should be included in the debugging prints.
 */
void setPrunFlag();

class Debugger {

    public:

        /**
         * Prints the start of the debugging output.
         */
        static void printStart();

        /**
         * Prints the end of the debugging output.
         */
        static void printEnd();

        /**
         * Prints a new line for output format.
         *
         * @param prun  if it's called from pruning
         */
        static void lb(bool prun);

        /**
         * Prints the coverage mask.
         *
         * @param coverageMask  the mask to be printed
         * @param originalMask  if it the original coverage mask
         * @param iteration     iteration of the coverage mask
         */
        static void printCoverageMask(const CoverageMask& coverageMask, bool originalMask,
                                      unsigned long iteration = 0);

        /**
         * Prints the quality score.
         *
         * @param bestScore the best quality score so far
         * @param score the current quality score
         */
        static void printQualityScores(float64 bestScore, float64 score) ;

        /**
         * Prints a rule. Called directly from pruning-irep or through printRule.
         *
         * @param conditionIterator iterator of the rules conditions
         * @param numConditions     number of conditions
         * @param head              the rules head
         */
        static void printRule(std::_List_const_iterator<Condition> conditionIterator, unsigned long numConditions,
                              const AbstractPrediction& head);

        /**
         * Prints a rule used to display alternative rules.
         *
         * @param condition
         * @param head
         */
        static void printRule(Refinement& condition, const AbstractPrediction& head);

        /**
         * Prints the number of pruned conditions.
         *
         * @param numPrunedConditions number of pruned conditions
         */
        static void printPrunedConditions(unsigned long numPrunedConditions);

        /**
         * Prints the distribution of sets.
         *
         * @param weights of the examples
         */
        static void printDistribution(const IWeightVector& weights);

        /**
         * Prints a matrix of which examples are covered by which feature.
         *
         * @param numLabels         number of labels
         * @param numStatistics     number of examples
         * @param uncoveredLabels   array of label coverage
         */
        static void printLabelCoverage(uint32 numLabels, uint32 numStatistics, float64* uncoveredLabels);

        /**
         * Prints the head score of the current refinement.
         *
         * @param headScore
         */
        static void printHeadScore(float64 headScore, bool final);

        /**
         * Prints if the algorithm should stop.
         *
         * @param shouldStop
         */
        static void printStopping(bool shouldStop);

        /**
         * Prints that a new rule has been induced.
         */
        static void printRuleInduction();

        /**
         * Prints the confusion Matrices used in rule_evaluation_label_wise.
         *
         *
         * @param confusionMatricesTotal    A pointer to a C-contiguous array of type `float64`, shape
         *                                  `(num_labels, NUM_CONFUSION_MATRIX_ELEMENTS)`, storing a confusion
         *                                  matrix that takes into account all examples for each label
         * @param confusionMatricesSubset   A pointer to a C-contiguous array of type `float64`, shape
         *                                  `(num_labels, NUM_CONFUSION_MATRIX_ELEMENTS)`, storing a confusion
         *                                  matrix that takes into account all all examples, which are covered by
         *                                  the previous refinement of the rule, for each label
         * @param confusionMatricesCovered  A pointer to a C-contiguous array of type `float64`, shape
         *                                  `(prediction.numPredictions_, NUM_CONFUSION_MATRIX_ELEMENTS)`, storing a
         *                                  confusion matrix that takes into account all examples, which are covered
         *                                  by the rule, for each label
         * @param numPredictions            Number of predictions which are evaluated. Usually all labels when
         *                                  called from find head, only the labels in the head when called from
         *                                  calculate prediction.
         * @param numMatrixElements:        Always four because of the evaluation metrics IN, IP, RN, RP.
         * @param uncovered                 False, if the confusion matrices in `confusion_matrices_covered`
         *                                  correspond to the examples that are covered by rule, True, if they
         *                                  correspond to the examples that are not covered by the rule.
         */
        static void printConfusionMatrices(const float64* confusionMatricesTotal,
                                           const float64* confusionMatricesSubset,
                                           const float64* confusionMatricesCovered,
                                           uint32 numPredictions,
                                           uint32 numMatrixElements,
                                           bool uncovered);

        /**
         * Prints the values for cin, cip, crn, crp, uin, uip, urn, urp.
         *
         * @param cin   The number of covered (C) labels that are irrelevant (I) according to the ground truth and
         *              for which the prediction in the rule's head is negative (N)
         * @param cip   The number of covered (C) labels that are irrelevant (I) according to the ground truth and
         *              for which the prediction in the rule's head is negative (N)
         * @param crn   The number of covered (C) labels that are relevant (R) according to the ground truth and for
         *              which the prediction in the rule's head is negative (N)
         * @param crp   The number of covered (C) labels that are relevant (R) according to the ground truth and for
         *              which the prediction in the rule's head is positive (P)
         * @param uin   The number of uncovered (U) labels that are irrelevant (I) according to the ground truth and
         *              for which the prediction in the rule's head is negative (N)
         * @param uip   The number of uncovered (U) labels that are irrelevant (I) according to the ground truth and
         *              for which the prediction in the rule's head is positive (P)
         * @param urn   The number of uncovered (U) labels that are relevant (R) according to the ground truth and
         *              for which the prediction in the rule's head is negative (N)
         * @param urp   The number of uncovered (U) labels that are relevant (R) according to the ground truth and
         *              for which the prediction in the rule's head is positive (P)
         * @param score score of the label given these metrics
         */
        static void printEvaluationConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp,
                                                   float64 uin, float64 uip, float64 urn, float64 urp, float64 score);

        /**
         * Prints "find head" when single head refinement find head is called. Useful to see the function of the
         * confusion matrices printed in printConfusionMatrices.
         */
        static void printFindHead();

        /**
         * Print "find refinement" when the refinement process is started.
         */
        static void printFindRefinement();

        /**
         * Prints "out of sample" to signal the confusion matrices printed in printConfusionMatrices are
         * out of sample.
         */
        static void printOutOfSample();

        /**
         * Prints "recalculate prediction internally" to signal the use of the confusion matrices printed in
         * printConfusionMatrices.
         */
        static void printRecalculateInternally();
};