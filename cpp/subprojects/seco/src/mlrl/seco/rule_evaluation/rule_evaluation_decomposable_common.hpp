/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/data/confusion_matrix.hpp"
#include "mlrl/seco/heuristics/heuristic.hpp"

namespace seco {

    /**
     * Calculates and returns the quality of a rule's prediction for a single output based on confusion matrices.
     *
     * @tparam StatisticType            The type of the elements that are stored in the confusion matrices
     * @param totalConfusionMatrix      A reference to an object of type `ConfusionMatrix` that takes into account all
     *                                  examples
     * @param coveredConfusionMatrix    A reference to an object of type `ConfusionMatrix` that takes into account all
     *                                  examples that are covered by the rule
     * @param heuristic                 The heuristic that should be used to assess the quality
     * @return                          The quality that has been calculated
     */
    template<typename StatisticType>
    static inline float32 calculateOutputWiseQuality(const ConfusionMatrix<StatisticType>& totalConfusionMatrix,
                                                     const ConfusionMatrix<StatisticType>& coveredConfusionMatrix,
                                                     const IHeuristic& heuristic) {
        const ConfusionMatrix<StatisticType> uncoveredConfusionMatrix = totalConfusionMatrix - coveredConfusionMatrix;
        return heuristic.evaluateConfusionMatrix(
          (float32) coveredConfusionMatrix.in, (float32) coveredConfusionMatrix.ip, (float32) coveredConfusionMatrix.rn,
          (float32) coveredConfusionMatrix.rp, (float32) uncoveredConfusionMatrix.in,
          (float32) uncoveredConfusionMatrix.ip, (float32) uncoveredConfusionMatrix.rn,
          (float32) uncoveredConfusionMatrix.rp);
    }

}
