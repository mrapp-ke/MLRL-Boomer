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
class AbstractPrediction : public IIndexVector {

    private:

        DenseVector<float64> predictedScoreVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        AbstractPrediction(uint32 numElements);

        typedef DenseVector<float64>::iterator score_iterator;

        typedef DenseVector<float64>::const_iterator score_const_iterator;

        /**
         * Returns a `score_iterator` to the beginning of the predicted scores.
         *
         * @return A `score_iterator` to the beginning
         */
        score_iterator scores_begin();

        /**
         * Returns a `score_iterator` to the end of the predicted scores.
         *
         * @return A `score_iterator` to the end
         */
        score_iterator scores_end();

        /**
         * Returns a `score_const_iterator` to the beginning of the predicted scores.
         *
         * @return A `score_const_iterator` to the beginning
         */
        score_const_iterator scores_cbegin() const;

        /**
         * Returns a `score_const_iterator` to the end of the predicted scores.
         *
         * @return A `score_const_iterator` to the end
         */
        score_const_iterator scores_cend() const;

        /**
         * Updates the given statistics by applying this prediction.
         *
         * @param statistics        A reference to an object of type `AbstractStatistics` to be updated
         * @param statisticIndex    The index of the statistic to be updated
         */
        virtual void apply(AbstractStatistics& statistics, uint32 statisticIndex) const = 0;

        /**
         * TODO
         */
        virtual void setNumElements(uint32 numElements);

        uint32 getNumElements() const override;

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
class FullPrediction : public AbstractEvaluatedPrediction {

    private:

        FullIndexVector indexVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        FullPrediction(uint32 numElements);

        typedef FullIndexVector::const_iterator index_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        void setNumElements(uint32 numElements) override;

        bool isPartial() const override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IStatisticsSubset> createSubset(const AbstractStatistics& statistics) const override;

        std::unique_ptr<IRuleRefinement> createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                              uint32 featureIndex) const override;

        std::unique_ptr<IHeadRefinement> createHeadRefinement(const IHeadRefinementFactory& factory) const override;

        void apply(AbstractStatistics& statistics, uint32 statisticIndex) const override;

};

/**
 * Stores the scores that are predicted by a rule that predicts for a subset of the available labels.
 */
class PartialPrediction : public AbstractEvaluatedPrediction {

    private:

        PartialIndexVector indexVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        PartialPrediction(uint32 numElements);

        typedef PartialIndexVector::iterator index_iterator;

        typedef PartialIndexVector::const_iterator index_const_iterator;

        /**
         * Returns an `index_iterator` to the beginning of the indices.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin();

        /**
         * Returns an `index_iterator` to the end of the indices.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the indices.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        void setNumElements(uint32 numElements) override;

        bool isPartial() const override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IStatisticsSubset> createSubset(const AbstractStatistics& statistics) const override;

        std::unique_ptr<IRuleRefinement> createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                              uint32 featureIndex) const override;

        std::unique_ptr<IHeadRefinement> createHeadRefinement(const IHeadRefinementFactory& factory) const override;

        void apply(AbstractStatistics& statistics, uint32 statisticIndex) const override;

};
