/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_vector_label_wise_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"


namespace boosting {

    /**
     * Calculates and returns a quality score that assesses the quality of the score that is predicted for a single
     * label, taking L1 and L2 regularization into account.
     *
     * @param score                     The predicted score
     * @param gradient                  The gradient
     * @param hessian                   The Hessian
     * @param l1RegularizationWeight    The weight of the L1 regularization
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The quality score that has been calculated
     */
    static inline float64 calculateLabelWiseQualityScore(float64 score, float64 gradient, float64 hessian,
                                                         float64 l1RegularizationWeight,
                                                         float64 l2RegularizationWeight) {
        float64 scorePow = score * score;
        float64 qualityScore =  (gradient * score) + (0.5 * hessian * scorePow);
        float64 l1RegularizationTerm = l1RegularizationWeight * std::abs(score);
        float64 l2RegularizationTerm = 0.5 * l2RegularizationWeight * scorePow;
        return qualityScore + l1RegularizationTerm + l2RegularizationTerm;
    }

}
