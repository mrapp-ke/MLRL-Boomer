/**
 * Provides classes representing the heads of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"


/**
 * Represents a potential head of a rule, whose quality is assessed in terms of a score.
 */
class HeadCandidate {

    public:

        /**
         * @param numPredictions    The number of labels for which the head predicts
         * @param labelIndices      A pointer to an array of type intp, shape `(numPredictions)`, representing the
         *                          indices of the labels for which the head predicts
         * @param predictedScores   A pointer to an array of type float64, shape `(numPredictions)`, representing the
         *                          scores that are predicted by the head
         * @param qualityScore      A score that assesses the quality of the head
         */
        HeadCandidate(intp numPredictions, intp* labelIndices, float64* predictedScores, float64 qualityScore);

        ~HeadCandidate();

        /**
         * The number of labels for which the head predicts.
         */
        intp numPredictions_;

        /**
         * A pointer to an array of type intp, shape `(numPredictions_)`, representing the indices of the labels for
         * which the head predicts
         */
        intp* labelIndices_;

        /**
         * A pointer to an array of type float64, shape `(numPredictions_)`, representing the scores that are predicted
         * by the head.
         */
        float64* predictedScores_;

        /**
         * A score that assesses the quality of the head.
         */
        float64 qualityScore_;

};
