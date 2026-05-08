/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/heuristics/heuristic.hpp"

namespace seco {

    /**
     * Calculates and returns the quality of a rule's prediction for a single output based on confusion matrices.
     *
     * @tparam StatisticType    The type of the elements that are stored in the confusion matrices
     * @param inTotal           The number of irrelevant labels predicted as negative by the current model
     * @param ipTotal           The number of irrelevant labels predicted as positive by the current model
     * @param rnTotal           The number of relevant labels predicted as negative by the current model
     * @param rpTotal           The number of relevant labels predicted as positive by the current model
     * @param inCovered         The number of irrelevant labels predicted as negative by a new rule
     * @param ipCovered         The number of irrelevant labels predicted as positive by a new rule
     * @param rnCovered         The number of relevant labels predicted as negative by a new rule
     * @param rpCovered         The number of relevant labels predicted as positive by a new rule
     * @param heuristic         The heuristic that should be used to assess the quality
     * @return                  The quality that has been calculated
     */
    template<typename StatisticType>
    static inline float32 calculateOutputWiseQuality(StatisticType inTotal, StatisticType ipTotal,
                                                     StatisticType rnTotal, StatisticType rpTotal,
                                                     StatisticType inCovered, StatisticType ipCovered,
                                                     StatisticType rnCovered, StatisticType rpCovered,
                                                     const IHeuristic& heuristic) {
        StatisticType inUncovered = inTotal - inCovered;
        StatisticType ipUncovered = ipTotal - ipCovered;
        StatisticType rnUncovered = rnTotal - rnCovered;
        StatisticType rpUncovered = rpTotal - rpCovered;

        return heuristic.evaluateConfusionMatrix((float32) inCovered, (float32) ipCovered, (float32) rnCovered,
                                                 (float32) rpCovered, (float32) inUncovered, (float32) ipUncovered,
                                                 (float32) rnUncovered, (float32) rpUncovered);
    }

}
