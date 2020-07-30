/**
 * Provides classes that store the predictions of rules, as well as corresponding quality scores.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"


namespace rule_evaluation {

    /**
     * Stores the default rule's predictions for each label.
     */
    class DefaultPrediction {

        public:

            /**
             * @param numPredictions    The number of labels for which the rule predicts
             * @param predictedScores   A pointer to an array of type float64, shape `(numPredictions)`, representing
             *                          the predicted scores
             */
            DefaultPrediction(intp numPredictions, float64* predictedScores);

            ~DefaultPrediction();

            /**
             * The number of labels for which the rule predicts.
             */
            intp numPredictions_;

            /**
             * A pointer to an array of type intp, shape `(numPredictions_)`, representing the predicted scores.
             */
            float64* predictedScores_;

    };

    /**
     * Assesses the overall quality of a rule's predictions for one or several labels.
     */
    class Prediction : public DefaultPrediction {

        public:

            /**
             * @param numPredictions        The number of labels for which the rule predicts
             * @param predictedScores       A pointer to an array of type float64, shape `(numPredictions)`,
             *                              representing the predicted scores
             * @param overallQualityScore   A score that assesses the overall quality of the predictions
             */
            Prediction(intp numPredictions, float64* predictedScores, float64 overallQualityScore);

            /**
             * A score that assesses the quality of the predictions.
             */
            float64 overallQualityScore_;

    };

    /**
     * Assesses the quality of a rule's predictions for one or several labels independently.
     */
    class LabelWisePrediction : public Prediction {

        public:

            /**
             * @param numPredictions        The number of labels for which the rule predicts
             * @param predictedScores       A pointer to an array of type float64, shape `(numPredictions)`,
             *                              representing the predicted scores
             * @param qualityScores         A pointer to an array of type float64, shape `(numPredictions)`,
             *                              representing the quality scores for individual labels
             * @param overallQualityScore   A score that assesses the overall quality of the predictions
             */
            LabelWisePrediction(intp numPredictions, float64* predictedScores, float64* qualityScores,
                                float64 overallQualityScore);

            ~LabelWisePrediction();

            /**
             * A pointer to an array of type float64, shape `(numPredictions_)`, representing the quality scores for
             * individual labels.
             */
            float64* qualityScores_;

    };

}
