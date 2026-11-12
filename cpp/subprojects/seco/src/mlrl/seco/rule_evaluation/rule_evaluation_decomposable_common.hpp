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
     * @param tp                The number of true positives
     * @param fp                The number of false positives
     * @param fn                The number of false negatives
     * @param tn                The number of true negatives
     * @param heuristic         The heuristic that should be used to assess the quality
     * @return                  The quality that has been calculated
     */
    template<typename StatisticType>
    static inline float32 calculateOutputWiseQuality(StatisticType tp, StatisticType fp, StatisticType fn,
                                                     StatisticType tn, const IHeuristic& heuristic) {
        return heuristic.evaluateConfusionMatrix((float32) tp, (float32) fp, (float32) fn, (float32) tn);
    }

}
