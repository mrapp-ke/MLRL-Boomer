/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_refinement/coverage_mask.hpp"
#include "mlrl/common/sampling/weight_vector_dense.hpp"
#include "mlrl/common/sampling/weight_vector_equal.hpp"
#include "mlrl/common/util/quality.hpp"

#include <memory>

// Forward declarations
class IStoppingCriterion;
class IStoppingCriterionFactory;
class IInstanceSampling;
class IClassificationInstanceSamplingFactory;
class IRowWiseLabelMatrix;
class IRegressionInstanceSamplingFactory;
class IRowWiseRegressionMatrix;
class IStatistics;
class IFeatureSubspace;
class IPrediction;
class IEvaluatedPrediction;
class IMarginalProbabilityCalibrationModel;
class IMarginalProbabilityCalibrator;
class IJointProbabilityCalibrationModel;
class IJointProbabilityCalibrator;

/**
 * Defines an interface for all classes that provide access to the indices of training examples that have been split
 * into a training set and a holdout set.
 */
class IPartition {
    public:

        virtual ~IPartition() {}

        /**
         * Creates and returns a new instance of the class `IStoppingCriterion`, based on the type of this partition.
         *
         * @param factory   A reference to an object of type `IStoppingCriterionFactory` that should be used to create
         *                  the instance
         * @return          An unique pointer to an object of type `IStoppingCriterion` that has been created
         */
        virtual std::unique_ptr<IStoppingCriterion> createStoppingCriterion(
          const IStoppingCriterionFactory& factory) = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSampling` that can be used in classification
         * problems, based on the type of this partition.
         *
         * @param factory           A reference to an object of type `IClassificationInstanceSamplingFactory` that
         *                          should be used to create the instance
         * @param labelMatrix       A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access
         *                          to the labels of individual training examples
         * @param statistics        A reference to an object of type `IStatistics` that provides access to the
         *                          statistics which serve as a basis for learning rules
         * @param exampleWeights    A reference to an object of type `EqualWeightVector` that provides access to the
         *                          weights of individual training examples
         * @return                  An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(
          const IClassificationInstanceSamplingFactory& factory, const IRowWiseLabelMatrix& labelMatrix,
          IStatistics& statistics, const EqualWeightVector& exampleWeights) = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSampling` that can be used in classification
         * problems, based on the type of this partition.
         *
         * @param factory           A reference to an object of type `IClassificationInstanceSamplingFactory` that
         *                          should be used to create the instance
         * @param labelMatrix       A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access
         *                          to the labels of individual training examples
         * @param statistics        A reference to an object of type `IStatistics` that provides access to the
         *                          statistics which serve as a basis for learning rules
         * @param exampleWeights    A reference to an object of type `DenseWeightVector<float32>` that provides access
         *                          to the weights of individual training examples
         * @return                  An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(
          const IClassificationInstanceSamplingFactory& factory, const IRowWiseLabelMatrix& labelMatrix,
          IStatistics& statistics, const DenseWeightVector<float32>& exampleWeights) = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSampling` that can be used in regression problems,
         * based on the type of this partition.
         *
         * @param factory           A reference to an object of type `IRegressionInstanceSamplingFactory` that should be
         *                          used to create the instance
         * @param regressionMatrix  A reference to an object of type `IRowWiseRegressionMatrix` that provides row-wise
         *                          access to the regression scores of individual training examples
         * @param statistics        A reference to an object of type `IStatistics` that provides access to the
         *                          statistics which serve as a basis for learning rules
         * @param exampleWeights    A reference to an object of type `EqualWeightVector` that provides access to the
         *                          weights of individual training examples
         * @return                  An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(
          const IRegressionInstanceSamplingFactory& factory, const IRowWiseRegressionMatrix& regressionMatrix,
          IStatistics& statistics, const EqualWeightVector& exampleWeights) = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSampling` that can be used in regression problems,
         * based on the type of this partition.
         *
         * @param factory           A reference to an object of type `IRegressionInstanceSamplingFactory` that should be
         *                          used to create the instance
         * @param regressionMatrix  A reference to an object of type `IRowWiseRegressionMatrix` that provides row-wise
         *                          access to the regression scores of individual training examples
         * @param statistics        A reference to an object of type `IStatistics` that provides access to the
         *                          statistics which serve as a basis for learning rules
         * @param exampleWeights    A reference to an object of type `DenseWeightVector<float64>` that provides access
         *                          to the weights of individual training examples
         * @return                  An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(
          const IRegressionInstanceSamplingFactory& factory, const IRowWiseRegressionMatrix& regressionMatrix,
          IStatistics& statistics, const DenseWeightVector<float32>& exampleWeights) = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of a rule's prediction for all examples
         * that do not belong to the current sample and are marked as covered according to a given object of type
         * `CoverageMask`.
         *
         * @param featureSubspace A reference to an object of type `IFeatureSubspace` that should be used to evaluate
         *                        the prediction
         * @param coverageMask    A reference to an object of type `CoverageMask` that keeps track of the examples that
         *                        are covered by the rule
         * @param head            A reference to an object of type `IPrediction` that stores the scores that are
         *                        predicted by the rule
         * @return                An object of type `Quality` that stores the calculated quality
         */
        virtual Quality evaluateOutOfSample(const IFeatureSubspace& featureSubspace, const CoverageMask& coverageMask,
                                            const IPrediction& head) = 0;

        /**
         * Recalculates and updates a rule's prediction based on all examples in the training set that are marked as
         * covered according to a given object of type `CoverageMask`.
         *
         * @param featureSubspace A reference to an object of type `IFeatureSubspace` that should be used to recalculate
         *                        the prediction
         * @param coverageMask    A reference to an object of type `CoverageMask` that keeps track of the examples that
         *                        are covered by the rule
         * @param headPtr         A reference to an unique pointer that stores an object of type `IEvaluatedPrediction`
         *                        to be updated
         */
        virtual void recalculatePrediction(const IFeatureSubspace& featureSubspace, const CoverageMask& coverageMask,
                                           std::unique_ptr<IEvaluatedPrediction>& headPtr) = 0;

        /**
         * Fits and returns a model for the calibration of marginal probabilities, based on the type of this partition.
         *
         * @param probabilityCalibrator A reference to an object of type `IMarginalProbabilityCalibrator` that should be
         *                              used to fit the calibration model
         * @param labelMatrix           A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise
         *                              access to the labels of the training examples
         * @param statistics            A reference to an object of type `IStatistics` that provides access to
         *                              statistics about the quality of predictions for training examples
         * @return                      An unique pointer to an object of type `IMarginalProbabilityCalibrationModel`
         *                              that has been fit
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
          const IMarginalProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
          const IStatistics& statistics) = 0;

        /**
         * Fits and returns a model for the calibration of joint probabilities, based on the type of this partition.
         *
         * @param probabilityCalibrator A reference to an object of type `IJointProbabilityCalibrator` that should be
         *                              used to fit the calibration model
         * @param labelMatrix           A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise
         *                              access to the labels of the training examples
         * @param statistics            A reference to an object of type `IStatistics` that provides access to
         *                              statistics about the quality of predictions for training examples
         * @return                      An unique pointer to an object of type `IJointProbabilityCalibrationModel` that
         *                              has been fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
          const IJointProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
          const IStatistics& statistics) = 0;
};
