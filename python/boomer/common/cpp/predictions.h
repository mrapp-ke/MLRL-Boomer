/**
 * Provides classes that store the predictions of rules, as well as corresponding quality scores.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"


/**
 * An abstract base class for all classes that store the scores that are predicted by a rule.
 */
class Prediction {

    public:

        /**
         * @param numPredictions The number of labels for which the rule predicts
         */
        Prediction(uint32 numPredictions);

        ~Prediction();

        /**
         * The number of labels for which the rule predicts.
         */
        uint32 numPredictions_;

        /**
         * A pointer to an array of type `uint32`, shape `(numPredictions_)`, representing the indices of the labels for
         * which the rule predicts or a null pointer, if the rule predicts for all labels.
         */
        uint32* labelIndices_;

        /**
         * A pointer to an array of type `float64`, shape `(numPredictions_)`, representing the predicted scores.
         */
        float64* predictedScores_;

};

/**
 * An abstract base class for all classes that store the scores that are predicted by a rule, as well as a quality score
 * that assesses the overall quality of the rule.
 */
class PredictionCandidate : public Prediction {

    public:

        /**
         * @param numPredictions The number of labels for which the rule predicts
         */
        PredictionCandidate(uint32 numPredictions);

        /**
         * A score that assesses the overall quality of the predictions.
         */
        float64 overallQualityScore_;

};
