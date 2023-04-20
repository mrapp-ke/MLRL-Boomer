/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/predictor.hpp"

/**
 * Defines an interface for all classes that allow to predict regression scores for given query examples.
 */
class IScorePredictor : public IPredictor<DensePredictionMatrix<float64>> {
    public:

        virtual ~IScorePredictor() override {};
};

/**
 * Defines an interface for all classes that allow to create instances of the type `IScorePredictor`.
 */
class IScorePredictorFactory {
    public:

        virtual ~IScorePredictorFactory() {};

        /**
         * Creates and returns a new object of the type `IScorePredictor`.
         *
         * @param featureMatrix                         A reference to an object of type `CsrConstView` that stores the
         *                                              feature values of the query examples to predict for
         * @param model                                 A reference to an object of type `RuleList` that should be used
         *                                              to obtain predictions
         * @param labelVectorSet                        A pointer to an object of type `LabelVectorSet` that stores all
         *                                              known label vectors or a null pointer, if no such set is
         *                                              available
         * @param marginalProbabilityCalibrationModel   A reference to an object of type
         *                                              `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                              calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel      A reference to an object of type
         *                                              `IJointProbabilityCalibrationModel` that may be used for the
         *                                              calibration of joint probabilities
         * @param numLabels                             The number of labels to predict for
         * @return                                      An unique pointer to an object of type `IScorePredictor` that
         *                                              has been created
         */
        virtual std::unique_ptr<IScorePredictor> create(
          const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
          const LabelVectorSet* labelVectorSet,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new object of the type `IScorePredictor`.
         *
         * @param featureMatrix                         A reference to an object of type `CsrConstView` that stores the
         *                                              feature values of the query examples to predict for
         * @param model                                 A reference to an object of type `RuleList` that should be used
         *                                              to obtain predictions
         * @param labelVectorSet                        A pointer to an object of type `LabelVectorSet` that stores all
         *                                              known label vectors or a null pointer, if no such set is
         *                                              available
         * @param marginalProbabilityCalibrationModel   A reference to an object of type
         *                                              `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                              calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel      A reference to an object of type
         *                                              `IJointProbabilityCalibrationModel` that may be used for the
         *                                              calibration of joint probabilities
         * @param numLabels                             The number of labels to predict for
         * @return                                      An unique pointer to an object of type `IScorePredictor` that
         *                                              has been created
         */
        virtual std::unique_ptr<IScorePredictor> create(
          const CsrConstView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a `IScorePredictor`.
 */
class IScorePredictorConfig : public IPredictorConfig<IScorePredictorFactory> {
    public:

        virtual ~IScorePredictorConfig() override {};
};
