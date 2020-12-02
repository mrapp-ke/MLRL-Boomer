/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../math/math.h"


namespace boosting {

    template<class ScoreVector, class GradientIterator, class HessianIterator>
    static inline void calculateLabelWisePredictionInternally(ScoreVector& scoreVector,
                                                              GradientIterator gradientIterator,
                                                              HessianIterator hessianIterator,
                                                              float64 l2RegularizationWeight) {
        uint32 numPredictions = scoreVector.getNumElements();
        typename ScoreVector::score_iterator scoreIterator = scoreVector.scores_begin();
        typename ScoreVector::quality_score_iterator qualityScoreIterator = scoreVector.quality_scores_begin();
        float64 overallQualityScore = 0;

        // For each label, calculate a score to be predicted, as well as a corresponding quality score...
        for (uint32 c = 0; c < numPredictions; c++) {
            float64 sumOfGradients = gradientIterator[c];
            float64 sumOfHessians =  hessianIterator[c];

            // Calculate the score to be predicted for the current label...
            float64 score = sumOfHessians + l2RegularizationWeight;
            score = score != 0 ? -sumOfGradients / score : 0;
            scoreIterator[c] = score;

            // Calculate the quality score for the current label...
            float64 scorePow = score * score;
            score = (sumOfGradients * score) + (0.5 * scorePow * sumOfHessians);
            qualityScoreIterator[c] = score + (0.5 * l2RegularizationWeight * scorePow);
            overallQualityScore += score;
        }

        // Add the L2 regularization term to the overall quality score...
        overallQualityScore += 0.5 * l2RegularizationWeight * l2NormPow(scoreIterator, numPredictions);
        scoreVector.overallQualityScore = overallQualityScore;
    }

}
