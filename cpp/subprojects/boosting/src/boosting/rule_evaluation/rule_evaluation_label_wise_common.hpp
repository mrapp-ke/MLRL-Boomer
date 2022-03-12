/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/math/math.hpp"


namespace boosting {

    /**
     * Returns the L1 regularization weight to be added to a specific gradient.
     *
     * @param gradient                  The gradient, the L1 regularization weight should be added to
     * @param l1RegularizationWeight    The L1 regularization weight
     * @return                          The L1 regularization weight to be added to the gradient
     */
    static inline constexpr float64 getL1RegularizationWeight(float64 gradient, float64 l1RegularizationWeight) {
        if (gradient > l1RegularizationWeight) {
            return -l1RegularizationWeight;
        } else if (gradient < -l1RegularizationWeight) {
            return l1RegularizationWeight;
        } else {
            return 0;
        }
    }

    /**
     * Calculates and returns the optimal score to be predicted for a single label, based on the corresponding gradient
     * and Hessian and taking L1 and L2 regularization into account.
     *
     * @param gradient                  The gradient that corresponds to the label
     * @param hessian                   The Hessian that corresponds to the label
     * @param l1RegularizationWeight    The weight of the L1 regularization
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The predicted score that has been calculated
     */
    static inline constexpr float64 calculateLabelWiseScore(float64 gradient, float64 hessian,
                                                            float64 l1RegularizationWeight,
                                                            float64 l2RegularizationWeight) {
        return divideOrZero(-gradient + getL1RegularizationWeight(gradient, l1RegularizationWeight),
                            hessian + l2RegularizationWeight);
    }

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

    /**
     * Calculates and returns a quality score that assesses the quality of the optimal prediction for a single label.
     *
     * @param gradient                  The gradient that corresponds to the label
     * @param hessian                   The Hessian that corresponds to the label
     * @param l1RegularizationWeight    The weight of the L1 regularization
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The quality score that has been calculated
     */
    static inline constexpr float64 calculateLabelWiseQualityScore(float64 gradient, float64 hessian,
                                                                   float64 l1RegularizationWeight,
                                                                   float64 l2RegularizationWeight) {
        float64 l1Weight = getL1RegularizationWeight(gradient, l1RegularizationWeight);
        float64 l1Term = l1Weight != 0
                            ? ((2 * gradient * l1Weight) - (3 * l1RegularizationWeight * l1RegularizationWeight))
                            : (-gradient * l1RegularizationWeight);
        return divideOrZero(-0.5 * (gradient * gradient + l1Term), hessian + l2RegularizationWeight);
    }

}
