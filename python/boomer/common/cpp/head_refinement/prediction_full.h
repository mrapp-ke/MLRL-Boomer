/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "prediction_evaluated.h"
#include "../indices/index_vector_full.h"

// Forward declarations
class IHistogram;


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

        void setNumElements(uint32 numElements, bool freeMemory) override;

        bool isPartial() const override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IStatisticsSubset> createSubset(const IHistogram& histogram) const override;

        std::unique_ptr<IRuleRefinement> createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                              uint32 featureIndex) const override;

        std::unique_ptr<IHeadRefinement> createHeadRefinement(const IHeadRefinementFactory& factory) const override;

        void apply(IStatistics& statistics, uint32 statisticIndex) const override;

};
