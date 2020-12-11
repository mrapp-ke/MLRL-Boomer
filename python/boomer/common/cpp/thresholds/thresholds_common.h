/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "thresholds.h"
#include "../input/feature_matrix.h"
#include "../input/nominal_feature_mask.h"
#include "../head_refinement/head_refinement_factory.h"
#include "../statistics/statistics.h"


/**
 * An entry that is stored in a cache and contains an unique pointer to a vector of arbitrary type. The field
 * `numConditions` specifies how many conditions the rule contained when the vector was updated for the last time. It
 * may be used to check if the vector is still valid or must be updated.
 *
 * @tparam T The type of the vector that is stored by the entry
 */
template<class T>
struct FilteredCacheEntry {
    FilteredCacheEntry<T>() : numConditions(0) { };
    std::unique_ptr<T> vectorPtr;
    uint32 numConditions;
};

static inline float64 evaluateOutOfSampleInternally(const IStatistics& statistics,
                                                    const IHeadRefinementFactory& headRefinementFactory,
                                                    const IWeightVector& weights, const CoverageMask& coverageMask,
                                                    const AbstractPrediction& prediction) {
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = prediction.createSubset(statistics);
    uint32 numStatistics = statistics.getNumStatistics();

    for (uint32 i = 0; i < numStatistics; i++) {
        if (weights.getWeight(i) == 0 && coverageMask.isCovered(i)) {
            statisticsSubsetPtr->addToSubset(i, 1);
        }
    }

    std::unique_ptr<IHeadRefinement> headRefinementPtr = prediction.createHeadRefinement(headRefinementFactory);
    const IScoreVector& scoreVector = headRefinementPtr->calculatePrediction(*statisticsSubsetPtr, false, false);
    return scoreVector.overallQualityScore;
}

static inline void recalculatePredictionInternally(const IStatistics& statistics,
                                                   const IHeadRefinementFactory& headRefinementFactory,
                                                   const CoverageMask& coverageMask, Refinement& refinement) {
    AbstractPrediction& head = *refinement.headPtr;
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = head.createSubset(statistics);
    uint32 numStatistics = statistics.getNumStatistics();

    for (uint32 i = 0; i < numStatistics; i++) {
        if (coverageMask.isCovered(i)) {
            statisticsSubsetPtr->addToSubset(i, 1);
        }
    }

    std::unique_ptr<IHeadRefinement> headRefinementPtr = head.createHeadRefinement(headRefinementFactory);
    const IScoreVector& scoreVector = headRefinementPtr->calculatePrediction(*statisticsSubsetPtr, false, false);
    scoreVector.updatePrediction(head);
}

static inline void updateStatisticsInternally(IStatistics& statistics, const CoverageMask& coverageMask,
                                              const AbstractPrediction& prediction) {
    uint32 numStatistics = statistics.getNumStatistics();

    for (uint32 i = 0; i < numStatistics; i++) {
        if (coverageMask.isCovered(i)) {
            prediction.apply(statistics, i);
        }
    }
}

/**
 * An abstract base class for all classes that provide access to thresholds that may be used by the first condition of a
 * rule that currently has an empty body and therefore covers the entire instance space.
 */
class AbstractThresholds : public IThresholds {

    protected:

        std::shared_ptr<IFeatureMatrix> featureMatrixPtr_;

        std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr_;

        std::shared_ptr<IStatistics> statisticsPtr_;

        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureMaskPtr     A shared pointer to an object of type `INominalFeatureMask` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsPtr             A shared pointer to an object of type `IStatistics` that provides access to
         *                                  statistics about the labels of the training examples
         * @param headRefinementFactoryPtr  A shared pointer to an object of type `IHeadRefinementFactory` that allows
         *                                  to create instances of the class that should be used to find the heads of
         *                                  rules
         */
        AbstractThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                           std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
                           std::shared_ptr<IStatistics> statisticsPtr,
                           std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr)
            : featureMatrixPtr_(featureMatrixPtr), nominalFeatureMaskPtr_(nominalFeatureMaskPtr),
              statisticsPtr_(statisticsPtr), headRefinementFactoryPtr_(headRefinementFactoryPtr) {

        }

        uint32 getNumExamples() const override {
            return featureMatrixPtr_->getNumExamples();
        }

        uint32 getNumFeatures() const override {
            return featureMatrixPtr_->getNumFeatures();
        }

        uint32 getNumLabels() const override {
            return statisticsPtr_->getNumLabels();
        }

};
