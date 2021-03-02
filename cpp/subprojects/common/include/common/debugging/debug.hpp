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
 * Sets the flag if the example weights should be included in the debugging prints.
 */
void setWeightsFlag();

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

class Debugger {

    public:

        static void printStart();

        static void printEnd();

        static void lb();

        static void printCoverageMask(const CoverageMask& coverageMask, bool originalMask,
                                      unsigned long iteration = 0);

        static void printQualityScores(float64 bestScore, float64 score) ;

        static void printRule(std::_List_const_iterator<Condition> conditionIterator, unsigned long numConditions,
                              const AbstractPrediction& head);

        static void printPrunedConditions(unsigned long numPrunedConditions);

        static void printWeights(const IWeightVector& weights);

        static void printLabelCoverage(uint32 numLabels, uint32 numStatistics, float64* uncoveredLabels);

        static void printHeadScore(float64 headScore);

        static void printStopping(bool shouldStop);

        static void printRuleInduction();
};