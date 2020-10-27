/**
 * Provides classes that store the predictions of rules, as well as corresponding quality scores.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"
#include "indices.h"


/**
 * An abstract base class for all classes that store the scores that are predicted by a rule.
 */
// TODO Rename to AbstractPrediction
class Prediction : public DenseVector<float64>, virtual public IIndexVector {

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        Prediction(uint32 numElements);

        ~Prediction();

        /**
         * A pointer to an array of type `uint32`, shape `(getNumElements())`, representing the indices of the labels
         * for which the rule predicts or a null pointer, if the rule predicts for all labels.
         */
        uint32* labelIndices_;

        /**
         * A pointer to an array of type `float64`, shape `(getNumElements())`, representing the predicted scores.
         */
        float64* predictedScores_;

        virtual uint32 getNumElements() const override = 0;

        virtual void setNumElements(uint32 numElements) override = 0;

};

/**
 * An abstract base class for all classes that store the scores that are predicted by a rule, as well as a quality score
 * that assesses the overall quality of the rule.
 */
// TODO Rename to AbstractEvaluatedPrediction
class PredictionCandidate : public Prediction {

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        PredictionCandidate(uint32 numElements);

        /**
         * A score that assesses the overall quality of the rule.
         */
        float64 overallQualityScore;

};

/**
 * Stores the scores that are predicted by a rule that predicts for all available labels.
 */
class FullPrediction : public PredictionCandidate, public RangeIndexVector {

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        FullPrediction(uint32 numElements);

        uint32 getNumElements() const override;

        void setNumElements(uint32 numElements) override;

};

/**
 * Stores the scores that are predicted by a rule that predicts for a subset of the available labels.
 */
class PartialPrediction : public PredictionCandidate, public DenseIndexVector {

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        PartialPrediction(uint32 numElements);

        uint32 getNumElements() const override;

        void setNumElements(uint32 numElements) override;

};
