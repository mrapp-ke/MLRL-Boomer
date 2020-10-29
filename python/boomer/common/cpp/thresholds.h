/**
 * Implements classes that provide access to the thresholds that may be used by the conditions of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "rule_refinement.h"
#include <unordered_map>


/**
 * Defines an interface for all classes that provide access a subset of thresholds that may be used by the conditions of
 * a rule with arbitrary body. The thresholds may include only those that correspond to the subspace of the instance
 * space that is covered by the rule.
 */
class IThresholdsSubset {

    public:

        virtual ~IThresholdsSubset() { };

        /**
         * Creates and returns a new instance of the type `IRuleRefinement` that allows to find the best refinement of
         * an existing rule that predicts for all available labels.
         *
         * @param labelIndices  A reference to an object of type `FullIndexVector` that provides access to the indices
         *                      of the labels for which the existing rule predicts
         * @param featureIndex  The index of the feature that should be considered when searching for refinements
         * @return              An unique pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual std::unique_ptr<IRuleRefinement> createRuleRefinement(const FullIndexVector& labelIndices,
                                                                      uint32 featureIndex) = 0;

        /**
         * Creates and returns a new instance of the type `IRuleRefinement` that allows to find the best refinement of
         * an existing rule that predicts for a subset of the available labels.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels for which the existing rule predicts
         * @param featureIndex  The index of the feature that should be considered when searching for refinements
         * @return              An unique pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& labelIndices,
                                                                      uint32 featureIndex) = 0;

        /**
         * Applies a refinement that has been found by an instance of the type `IRuleRefinement`, which was previously
         * created via the function `createRuleRefinement`.
         *
         * This causes the thresholds that will be available for further refinements to be filtered such that only those
         * thresholds that correspond to the subspace of the instance space that is covered by the refined rule are
         * included.
         *
         * @param refinement A reference to an object of type `Refinement`, representing the refinement to be applied
         */
        virtual void applyRefinement(Refinement& refinement) = 0;

        /**
         * Recalculates the scores to be predicted by a refinement that has been found by an instance of the type
         * `IRuleRefinement`, which was previously created via the function `createRuleRefinement`, and updates the head
         * of the refinement accordingly.
         *
         * When calculating the updated scores the weights of the individual training examples are ignored and equally
         * distributed weights are assumed instead.
         *
         * @param refinement A reference to an object of type `Refinement`, whose head should be updated
         */
        virtual void recalculatePrediction(Refinement& refinement) const = 0;

        /**
         * Applies the predictions of a rule to the statistics that correspond to the current subset.
         *
         * @param prediction A reference to an object of type `AbstractPrediction`, representing the predictions to be
         *                   applied
         */
        virtual void applyPrediction(const AbstractPrediction& prediction) = 0;

};

/**
 * An abstract base class for all classes that provide access to thresholds that may be used by the first condition of a
 * rule that currently has an empty body and therefore covers the entire instance space.
 */
class AbstractThresholds : virtual public IMatrix {

    protected:

        std::shared_ptr<IFeatureMatrix> featureMatrixPtr_;

        std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr_;

        std::shared_ptr<AbstractStatistics> statisticsPtr_;

        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureVectorPtr   A shared pointer to an object of type `INominalFeatureVector` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsPtr             A shared pointer to an object of type `AbstractStatistics` that provides
         *                                  access to statistics about the labels of the training examples
         * @param headRefinementFactoryPtr  A shared pointer to an object of type `IHeadRefinementFactory` that allows
         *                                  to create instances of the class that should be used to find the heads of
         *                                  rules
         */
        AbstractThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                           std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                           std::shared_ptr<AbstractStatistics> statisticsPtr,
                           std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr);

        /**
         * Creates and returns a new subset of the thresholds, which initially contains all of the thresholds.
         *
         * @param weights   A reference to an object of type `IWeightVector` that provides access to the weights of the
         *                  individual training examples
         * @return          An unique pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual std::unique_ptr<IThresholdsSubset> createSubset(const IWeightVector& weights) = 0;

        /**
         * Returns the total number of available labels.
         *
         * @return The total number of available labels
         */
        uint32 getNumLabels() const;

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

};

/**
 * A wrapper for a (filtered) feature vector that is stored in the cache. The field `numConditions` specifies how many
 * conditions the rule contained when the array was updated for the last time. It may be used to check if the array is
 * still valid or must be updated.
 */
struct CacheEntry {
    CacheEntry() : numConditions(0) { };
    std::unique_ptr<FeatureVector> featureVectorPtr;
    uint32 numConditions;
};

/**
 * Provides access to all thresholds that result from the feature values of the training examples.
 */
class ExactThresholdsImpl : public AbstractThresholds {

    private:

        // Forward declarations
        class ThresholdsSubset;

        std::unordered_map<uint32, std::unique_ptr<FeatureVector>> cache_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureVectorPtr   A shared pointer to an object of type `INominalFeatureVector` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsPtr             A shared pointer to an object of type `AbstractStatistics` that provides
         *                                  access to statistics about the labels of the training examples
         * @param headRefinementFactoryPtr  A shared pointer to an object of type `IHeadRefinementFactory` that allows
         *                                  to create instances of the class that should be used to find the heads of
         *                                  rules
         */
        ExactThresholdsImpl(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                            std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                            std::shared_ptr<AbstractStatistics> statisticsPtr,
                            std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr);

        std::unique_ptr<IThresholdsSubset> createSubset(const IWeightVector& weights) override;

};

/**
 * Provides access to the thresholds that result from applying a binning method to the feature values of the training
 * examples.
 */
class ApproximateThresholdsImpl : public AbstractThresholds {

    private:

        // Forward declarations
        class ThresholdsSubset;

        /**
         * A wrapper for statistics and bins that is stored in the cache.
         */
        struct BinCacheEntry {
            std::unique_ptr<AbstractStatistics> statisticsPtr;
            std::unique_ptr<BinVector> binVectorPtr;
        };

        std::shared_ptr<IBinning> binningPtr_;

        uint32 numBins_;

        std::unordered_map<uint32, BinCacheEntry> cache_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureVectorPtr   A shared pointer to an object of type `INominalFeatureVector` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsPtr             A shared pointer to an object of type `AbstractStatistics` that provides
         *                                  access to statistics about the labels of the training examples
         * @param headRefinementFactoryPtr  A shared pointer to an object of type `IHeadRefinementFactory` that allows
         *                                  to create instances of the class that should be used to find the heads of
         *                                  rules
         * @param binningPtr                A shared pointer to an object of type `IBinning` that implements the binning
         *                                  method to be used
         * @param numBins                   The number of bins that should be used by the given binning method
         */
        ApproximateThresholdsImpl(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                  std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                                  std::shared_ptr<AbstractStatistics> statisticsPtr,
                                  std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
                                  std::shared_ptr<IBinning> binningPtr, uint32 numBins);

        std::unique_ptr<IThresholdsSubset> createSubset(const IWeightVector& weights) override;

};
