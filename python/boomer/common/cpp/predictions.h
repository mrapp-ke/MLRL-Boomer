/**
 * Provides classes that store the predictions of rules, as well as corresponding quality scores.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"
#include "indices.h"

// Forward declarations
class AbstractStatistics;


/**
 * An abstract base class for all classes that store the scores that are predicted by a rule.
 */
class AbstractPrediction : public DenseVector<float64>, virtual public IIndexVector {

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        AbstractPrediction(uint32 numElements);

        /**
         * Updates the given statistics by applying this prediction.
         *
         * @param statistics        A reference to an object of type `AbstractStatistics` to be updated
         * @param statisticIndex    The index of the statistic to be updated
         */
        virtual void apply(AbstractStatistics& statistics, uint32 statisticIndex) const = 0;

        virtual uint32 getNumElements() const override = 0;

        virtual void setNumElements(uint32 numElements) = 0;

};

/**
 * An abstract base class for all classes that store the scores that are predicted by a rule, as well as a quality score
 * that assesses the overall quality of the rule.
 */
class AbstractEvaluatedPrediction : public AbstractPrediction {

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        AbstractEvaluatedPrediction(uint32 numElements);

        /**
         * A score that assesses the overall quality of the rule.
         */
        float64 overallQualityScore;

};

/**
 * Stores the scores that are predicted by a rule that predicts for all available labels.
 */
class FullPrediction : public AbstractEvaluatedPrediction, public FullIndexVector {

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        FullPrediction(uint32 numElements);

        uint32 getNumElements() const override;

        void setNumElements(uint32 numElements) override;

        void apply(AbstractStatistics& statistics, uint32 statisticIndex) const override;

};

/**
 * Stores the scores that are predicted by a rule that predicts for a subset of the available labels.
 */
class PartialPrediction : public AbstractEvaluatedPrediction, public PartialIndexVector {

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        PartialPrediction(uint32 numElements);

        uint32 getNumElements() const override;

        void setNumElements(uint32 numElements) override;

        void apply(AbstractStatistics& statistics, uint32 statisticIndex) const override;

};
