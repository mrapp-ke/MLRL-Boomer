#include "common/pruning/pruning_irep.hpp"

#include "../head_refinement/prediction.h"
#include "../head_refinement/prediction_partial.h"
#include "../debugging/global.h"
#include <iostream>


std::unique_ptr<CoverageMask> IREP::prune(IThresholdsSubset& thresholdsSubset, const IPartition& partition,
                                          ConditionList& conditions, const AbstractPrediction& head) const {
    ConditionList::size_type numConditions = conditions.getNumConditions();
    std::unique_ptr<CoverageMask> bestCoverageMaskPtr;

    // Only rules with more than one condition can be pruned...
    if (numConditions > 1) {
        // Calculate the quality score of the original rule on the prune set...
        const CoverageMask& originalCoverageMask = thresholdsSubset.getCoverageMask();
        float64 bestQualityScore = partition.evaluateOutOfSample(thresholdsSubset, originalCoverageMask, head);

        // printing of the coverage mask
        // TODO: usage of debug mode

        // test for the debug mode
        std::cout << (debug_flag == 1 ? "debugging enabled\n" : "debugging not enabled\n");

        std::cout << "\nthe original coverage mask:\n";
        for (uint32 i = 0; i < originalCoverageMask.getNumElements(); i++) {
            std::cout << "index " << i <<
                (originalCoverageMask.isCovered(i) ? " covered" : " not covered") << "\n";
        }

        // Create a copy of the original coverage mask...
        bestCoverageMaskPtr = std::make_unique<CoverageMask>(originalCoverageMask);

        // Reset the given thresholds...
        thresholdsSubset.resetThresholds();

        // We process the existing rule's conditions (except for the last one) in the order they have been learned. At
        // each iteration, we calculate the quality score of a rule that only contains the conditions processed so far
        // and keep track of the best rule...
        ConditionList::const_iterator conditionIterator = conditions.cbegin();
        ConditionList::size_type numPrunedConditions = 0;

        for (std::list<Condition>::size_type n = 1; n < numConditions; n++) {
            // Filter the thresholds by applying the current condition...
            const Condition& condition = *conditionIterator;
            thresholdsSubset.filterThresholds(condition);

            //printing the rule
            ConditionList::const_iterator& printConditionIterator = conditionIterator;
            std::cout << "\n" << "{";
            for(std::list<Condition>::size_type n = 1; n <= numConditions; n++) {
                uint32 comp = static_cast<uint32>(printConditionIterator->comparator);
                std::cout << printConditionIterator->featureIndex <<
                    " "<< (comp == 0 ? "<=" : comp == 1 ? ">" : comp == 2 ? "==" : "!=") <<
                    " " << printConditionIterator->threshold << (n == numConditions ? "" : ", ");
                printConditionIterator++;
            }
            std::cout << "} -> ";

            if (head.isPartial()) {
                const PartialPrediction* pred = dynamic_cast<const PartialPrediction*>(&head);
                for (uint32 i = 0; i < head.getNumElements(); i++) {
                    std::cout << "(" << i << " = " << pred->indices_cbegin()[i] <<
                        (i + 1 == head.getNumElements() ? "" : ", ");
                }
            }
            std::cout << ")\n";

            // Calculate the quality score of a rule that contains the conditions that have been processed so far...
            const CoverageMask& coverageMask = thresholdsSubset.getCoverageMask();
            float64 qualityScore = partition.evaluateOutOfSample(thresholdsSubset, coverageMask, head);

            // printing of the iteration coverage mask
            std::cout << "\nthe " << n << ". coverage mask:\n";
            for (uint32 i = 0; i < coverageMask.getNumElements(); i++) {
                std::cout << "index " << i <<
                          (coverageMask.isCovered(i) ? " covered" : " not covered") << "\n";
            }
            // printing the quality scores
            std::cout << "\nbest quality score: " << bestQualityScore << "\n";
            std::cout << "current quality score " << qualityScore << "\n";

            // Check if the quality score is better than the best quality score known so far (reaching the same score
            // with fewer conditions is considered an improvement)...
            if (qualityScore < bestQualityScore || (numPrunedConditions == 0 && qualityScore == bestQualityScore)) {
                bestQualityScore = qualityScore;
                bestCoverageMaskPtr = std::make_unique<CoverageMask>(coverageMask);
                numPrunedConditions = (numConditions - n);
            }

            std::cout << "number of conditions to prune: " << numPrunedConditions << "\n\n";

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
