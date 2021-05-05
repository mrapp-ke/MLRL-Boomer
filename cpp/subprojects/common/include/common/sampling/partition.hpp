/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <memory>

// Forward declarations
class IInstanceSubSampling;
class IInstanceSubSamplingFactory;
class ILabelMatrix;
class IThresholdsSubset;
class ICoverageState;
class Refinement;
class AbstractPrediction;


/**
 * Defines an interface for all classes that provide access to the indices of training examples that have been split
 * into a training set and a holdout set.
 */
class IPartition {

    public:

        virtual ~IPartition() { };

        /**
         * Creates and returns a new instance of the class `IInstanceSubSampling`, based on the type of this partition
         * matrix.
         *
         * @param factory       A reference to an object of type `IInstanceSubSamplingFactory` that should be used to
         *                      create the instance
         * @param labelMatrix   A reference to an object of type `ILabelMatrix` that provides access to the labels of
         *                      the training examples
         * @return              An unique pointer to an object of type `IInstanceSubSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSubSampling> createInstanceSubSampling(
            const IInstanceSubSamplingFactory& factory, const ILabelMatrix& labelMatrix) = 0;

        /**
         * Calculates and returns a quality score that assesses the quality of a rule's prediction for all examples that
         * do not belong to the current sub-sample and are marked as covered according to a given object of type
         * `ICoverageState`.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          evaluate the prediction
         * @param coverageState     A reference to an object of type `ICoverageState` that keeps track of the examples
         *                          that are covered by the rule
         * @param head              A reference to an object of type `AbstractPrediction` that stores the scores that
         *                          are predicted by the rule
         * @return                  The calculated quality score
         */
        virtual float64 evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset,
                                            const ICoverageState& coverageState,
                                            const AbstractPrediction& head) = 0;

        /**
         * Recalculates the scores to be predicted by a refinement based on all examples in the training set that are
         * marked as covered according to a given object of type `ICoverageState` and updates the head of the refinement
         * accordingly.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          recalculate the scores
         * @param coverageState     A reference to an object of type `ICoverageState` that keeps track of the examples
         *                          that are covered by the refinement
         * @param refinement        A reference to an object of type `Refinement`, whose head should be updated
         */
        virtual void recalculatePrediction(const IThresholdsSubset& thresholdsSubset,
                                           const ICoverageState& coverageState, Refinement& refinement) = 0;

};
