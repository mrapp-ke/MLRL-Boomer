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
     * @param uin               The number of uncovered (U), irrelevant (I) labels for which the rule predicts
     *                          negatively (N)
     * @param uip               The number of uncovered (U), irrelevant (I) labels for which the rule predicts
     *                          positively (P)
     * @param urn               The number of uncovered (U), relevant (R) labels for which the rule predicts negatively
     *                          (N)
     * @param urp               The number of uncovered (U), relevant (R) labels for which the rule predicts positively
     *                          (P)
     * @param cin               The number of covered (C), irrelevant (I) labels for which the rule predicts negatively
     *                          (N)
     * @param cip               The number of covered (C), irrelevant (I) labels for which the rule predicts positively
     *                          (P)
     * @param crn               The number of covered (C), relevant (R) labels for which the rule predicts negatively
     *                          (N)
     * @param crp               The number of covered (C), relevant (R) labels for which the rule predicts positively
     *                          (P)
     * @param heuristic         The heuristic that should be used to assess the quality
     * @return                  The quality that has been calculated
     */
    template<typename StatisticType>
    static inline float32 calculateOutputWiseQuality(StatisticType uin, StatisticType uip, StatisticType urn,
                                                     StatisticType urp, StatisticType cin, StatisticType cip,
                                                     StatisticType crn, StatisticType crp,
                                                     const IHeuristic& heuristic) {
        float32 tp = (float32) cin + (float32) crp;
        float32 fp = (float32) cip + (float32) crn;
        float32 fn = (float32) uin + (float32) urp;
        float32 tn = (float32) uip + (float32) urn;
        return heuristic.evaluateConfusionMatrix(tp, fp, fn, tn);
    }

}
