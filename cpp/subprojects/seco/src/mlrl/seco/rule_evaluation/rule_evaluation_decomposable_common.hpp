/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/data/confusion_matrix.hpp"
#include "mlrl/seco/heuristics/heuristic.hpp"

namespace seco {

    /**
     * Calculates and returns the quality of a rule's prediction for a single output.
     *
     * @param totalConfusionMatrix      A reference to an object of type `ConfusionMatrix` that takes into account all
     *                                  examples
     * @param coveredConfusionMatrix    A reference to an object of type `ConfusionMatrix` that takes into account all
     *                                  examples that are covered by the rule
     * @param heuristic                 The heuristic that should be used to assess the quality
     * @return                          The quality that has been calculated
     */
    static inline float64 calculateOutputWiseQuality(const ConfusionMatrix<uint32>& totalConfusionMatrix,
                                                     const ConfusionMatrix<uint32>& coveredConfusionMatrix,
                                                     const IHeuristic& heuristic) {
        const ConfusionMatrix<uint32> uncoveredConfusionMatrix = totalConfusionMatrix - coveredConfusionMatrix;
        return heuristic.evaluateConfusionMatrix(coveredConfusionMatrix.in, coveredConfusionMatrix.ip,
                                                 coveredConfusionMatrix.rn, coveredConfusionMatrix.rp,
                                                 uncoveredConfusionMatrix.in, uncoveredConfusionMatrix.ip,
                                                 uncoveredConfusionMatrix.rn, uncoveredConfusionMatrix.rp);
    }

}
