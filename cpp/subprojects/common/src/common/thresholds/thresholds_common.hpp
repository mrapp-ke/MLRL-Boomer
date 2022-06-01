/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/thresholds/thresholds.hpp"
#include "common/input/feature_matrix.hpp"
#include "common/input/nominal_feature_mask.hpp"
#include "omp.h"


template<typename IndexIterator, typename WeightVector>
static inline float64 evaluateOutOfSampleInternally(IndexIterator indexIterator, uint32 numExamples,
                                                    const WeightVector& weights, const CoverageMask& coverageMask,
                                                    const IStatistics& statistics,
                                                    const AbstractPrediction& prediction) {
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = prediction.createStatisticsSubset(statistics);

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indexIterator[i];

        if (weights.getWeight(exampleIndex) == 0 && coverageMask.isCovered(exampleIndex)) {
            statisticsSubsetPtr->addToSubset(exampleIndex, 1);
        }
    }

    const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();
    return scoreVector.overallQualityScore;
}

template<typename WeightVector>
static inline float64 evaluateOutOfSampleInternally(const WeightVector& weights, const CoverageSet& coverageSet,
                                                    const IStatistics& statistics,
                                                    const AbstractPrediction& prediction) {
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = prediction.createStatisticsSubset(statistics);
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator iterator = coverageSet.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = iterator[i];

        if (weights.getWeight(exampleIndex) == 0) {
            statisticsSubsetPtr->addToSubset(exampleIndex, 1);
        }
    }

    const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();
    return scoreVector.overallQualityScore;
}

template<typename WeightVector>
static inline float64 evaluateOutOfSampleInternally(const WeightVector& weights, const CoverageSet& coverageSet,
                                                    BiPartition& partition, const IStatistics& statistics,
                                                    const AbstractPrediction& prediction) {
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = prediction.createStatisticsSubset(statistics);
    const BitVector& holdoutSet = partition.getSecondSet();
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator iterator = coverageSet.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = iterator[i];

        if (weights.getWeight(exampleIndex) == 0 && holdoutSet[exampleIndex]) {
            statisticsSubsetPtr->addToSubset(exampleIndex, 1);
        }
    }

    const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();
    return scoreVector.overallQualityScore;
}

template<typename IndexIterator>
static inline void recalculatePredictionInternally(IndexIterator indexIterator, uint32 numExamples,
                                                   const CoverageMask& coverageMask, const IStatistics& statistics,
                                                   Refinement& refinement) {
    AbstractPrediction& head = *refinement.headPtr;
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = head.createStatisticsSubset(statistics);

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indexIterator[i];

        if (coverageMask.isCovered(exampleIndex)) {
            statisticsSubsetPtr->addToSubset(exampleIndex, 1);
        }
    }

    const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();
    scoreVector.updatePrediction(head);
}

static inline void recalculatePredictionInternally(const CoverageSet& coverageSet, const IStatistics& statistics,
                                                   Refinement& refinement) {
    AbstractPrediction& head = *refinement.headPtr;
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = head.createStatisticsSubset(statistics);
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator iterator = coverageSet.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = iterator[i];
        statisticsSubsetPtr->addToSubset(exampleIndex, 1);
    }

    const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();
    scoreVector.updatePrediction(head);
}

static inline void recalculatePredictionInternally(const CoverageSet& coverageSet, BiPartition& partition,
                                                   const IStatistics& statistics, Refinement& refinement) {
    AbstractPrediction& head = *refinement.headPtr;
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = head.createStatisticsSubset(statistics);
    const BitVector& holdoutSet = partition.getSecondSet();
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator iterator = coverageSet.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = iterator[i];

        if (holdoutSet[exampleIndex]) {
            statisticsSubsetPtr->addToSubset(exampleIndex, 1);
        }
    }

    const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();
    scoreVector.updatePrediction(head);
}

/**
 * An abstract base class for all classes that provide access to thresholds that may be used by the first condition of a
 * rule that currently has an empty body and therefore covers the entire instance space.
 */
class AbstractThresholds : public IThresholds {

    protected:

        /**
         * A reference to an object of type `IColumnWiseFeatureMatrix` that provides column-wise access to the feature
         * values of the training examples.
         */
        const IColumnWiseFeatureMatrix& featureMatrix_;

        /**
         * A reference to an object of type `INominalFeatureMask` that provides access to the information whether
         * individual feature are nominal or not.
         */
        const INominalFeatureMask& nominalFeatureMask_;

        /**
         * A reference to an object of type `IStatisticsProvider` that provides access to statistics about the labels of
         * the training examples.
         */
        IStatisticsProvider& statisticsProvider_;

    public:

        /**
         * @param featureMatrix         A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                              column-wise access to the feature values of individual training examples
         * @param nominalFeatureMask    A reference  to an object of type `INominalFeatureMask` that provides access to
         *                              the information whether individual features are nominal or not
         * @param statisticsProvider    A reference to an object of type `IStatisticsProvider` that provides access to
         *                              statistics about the labels of the training examples
         */
        AbstractThresholds(const IColumnWiseFeatureMatrix& featureMatrix, const INominalFeatureMask& nominalFeatureMask,
                           IStatisticsProvider& statisticsProvider)
            : featureMatrix_(featureMatrix), nominalFeatureMask_(nominalFeatureMask),
              statisticsProvider_(statisticsProvider) {

        }

        virtual ~AbstractThresholds() override { };

        IStatisticsProvider& getStatisticsProvider() const override final {
            return statisticsProvider_;
        }

};
