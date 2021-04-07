#include <iostream>
#include "common/pruning/pruning_irep.hpp"

#include "common/head_refinement/prediction.hpp"
#include "common/debugging/debug.hpp"


std::unique_ptr<CoverageMask> IREP::prune(IThresholdsSubset& thresholdsSubset, const IPartition& partition,
                                          ConditionList& conditions, const AbstractPrediction& head) const {
    ConditionList::size_type numConditions = conditions.getNumConditions();
    std::unique_ptr<CoverageMask> bestCoverageMaskPtr;

    // Only rules with more than one condition can be pruned...
    if (numConditions > 1) {

        Debugger::lb();

        // Calculate the quality score of the original rule on the prune set...
        const CoverageMask& originalCoverageMask = thresholdsSubset.getCoverageMask();
        float64 bestQualityScore = partition.evaluateOutOfSample(thresholdsSubset, originalCoverageMask, head);

        // Debugger: print the original coverage mask
        Debugger::printCoverageMask(originalCoverageMask, true);

        // Create a copy of the original coverage mask...
        bestCoverageMaskPtr = std::make_unique<CoverageMask>(originalCoverageMask);

        // Reset the given thresholds...
        thresholdsSubset.resetThresholds();

        // We process the existing rule's conditions (except for the last one) in the order they have been learned. At
        // each iteration, we calculate the quality score of a rule that only contains the conditions processed so far
        // and keep track of the best rule...
        auto conditionIterator = conditions.cbegin();
        ConditionList::size_type numPrunedConditions = 0;

        for (std::list<Condition>::size_type n = 1; n < numConditions; n++) {
            // Filter the thresholds by applying the current condition...
            const Condition& condition = *conditionIterator;
            thresholdsSubset.filterThresholds(condition);


            // Debugger: print rule
            Debugger::printRule(conditions.cbegin(), numConditions, head);

            // Calculate the quality score of a rule that contains the conditions that have been processed so far...
            const CoverageMask& coverageMask = thresholdsSubset.getCoverageMask();
            float64 qualityScore = partition.evaluateOutOfSample(thresholdsSubset, coverageMask, head);

            // Debugger: print iteration mask
            Debugger::printCoverageMask(originalCoverageMask, false, n);
            // Debugger: print quality scores
            Debugger::printQualityScores(bestQualityScore, qualityScore);

            // Check if the quality score is better than the best quality score known so far (reaching the same score
            // with fewer conditions is considered an improvement)...
            if (qualityScore < bestQualityScore || (numPrunedConditions == 0 && qualityScore == bestQualityScore)) {
                bestQualityScore = qualityScore;
                bestCoverageMaskPtr = std::make_unique<CoverageMask>(coverageMask);
                numPrunedConditions = (numConditions - n);
            }

            // Debugger: print number of pruned conditions
            Debugger::printPrunedConditions(numPrunedConditions);

            conditionIterator++;
        }

        // Remove the pruned conditions...
        while (numPrunedConditions > 0) {
            conditions.removeLast();
            numPrunedConditions--;
        }
    }

    return bestCoverageMaskPtr;
}
