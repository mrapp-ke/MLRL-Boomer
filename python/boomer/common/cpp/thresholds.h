/**
 * Implements classes that provide access to the thresholds that may be used by the conditions of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "data.h"
#include "input_data.h"
#include "sub_sampling.h"
#include "rule_refinement.h"
#include <memory>

/**
 * Defines an interface for all classes that provide access a subset of thresholds that may be used by the conditions of
 * a rule with arbitrary body. The thresholds may include on those that correspond to the subspace of the instance space
 * that is covered by the rule.
 */
class IThresholdsSubset {

    public:

        virtual ~IThresholdsSubset() { };

        /**
         * Creates and returns a new instance of the type `IRuleRefinement` that allows to find the best refinement of
         * an existing rule, which results from adding a new condition that corresponds to the feature at a specific
         * index.
         *
         * @param featureIndex  The index of the feature, the new condition corresponds to
         * @param numConditions The number of conditions of the existing rule. This is used to check if caches are still
         *                      valid
         * @return              A pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual IRuleRefinement* createRuleRefinement(uint32 featureIndex, uint32 numConditions) = 0;

};

/**
 * An abstract base class for all classes that provide access to thresholds that may be used by the first condition of a
 * rule that currently has an empty body and therefore covers the entire instance space.
 */
class AbstractThresholds : virtual public IMatrix {

    private:

        std::shared_ptr<IFeatureMatrix> featureMatrixPtr_;

        std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr_;

        std::shared_ptr<AbstractStatistics> statisticsPtr_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureVectorPtr   A shared pointer to an object of type `INominalFeatureVector` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsPtr             A shared pointer to an object of type `AbstractStatistics` that provides
         *                                  access to statistics about the labels of the training examples
         */
        AbstractThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                           std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                           std::shared_ptr<AbstractStatistics> statisticsPtr);

        /**
         * Creates and returns a new subset of the thresholds, which initially contains all of the thresholds.
         *
         * @param weights   A pointer to an object of type `IWeightVector` that provides access to the weights of the
         *                  individual training examples
         * @return          A pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual IThresholdsSubset* createSubset(IWeightVector* weights) = 0;

        uint32 getNumRows() override;

        uint32 getNumCols() override;

};
