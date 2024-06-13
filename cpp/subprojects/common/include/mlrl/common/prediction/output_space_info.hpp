/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csr.hpp"

#include <memory>

class IJointProbabilityCalibrator;
class IJointProbabilityCalibratorFactory;
class RuleList;
class IMarginalProbabilityCalibrationModel;
class IJointProbabilityCalibrationModel;
class IBinaryPredictor;
class IBinaryPredictorFactory;
class ISparseBinaryPredictor;
class ISparseBinaryPredictorFactory;
class IScorePredictor;
class IScorePredictorFactory;
class IProbabilityPredictor;
class IProbabilityPredictorFactory;

/**
 * Defines an interface for all classes that provide information about the output space that may be used as a basis for
 * making predictions.
 */
class MLRLCOMMON_API IOutputSpaceInfo {
    public:

        virtual ~IOutputSpaceInfo() {}

        /**
         * Creates and returns a new instance of the class `IJointProbabilityCalibrator`, based on the type of this
         * information about the output space.
         *
         * @param factory                             A reference to an object of type
         *                                            `IJointProbabilityCalibratorFactory` that should be used to create
         *                                            the instance
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @return                                    An unique pointer to an object of type
         *                                            `IJointProbabilityCalibrator` that has been created
         */
        virtual std::unique_ptr<IJointProbabilityCalibrator> createJointProbabilityCalibrator(
          const IJointProbabilityCalibratorFactory& factory,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const = 0;

        /**
         * Creates and returns a new instance of the class `IBinaryPredictor`, based on the type of this information
         * about the output space.
         *
         * @param factory                             A reference to an object of type `IBinaryPredictorFactory` that
         *                                            should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CContiguousView` that provides
         *                                            row-wise access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IBinaryPredictor` that has
         *                                            been created
         */
        virtual std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IBinaryPredictor`, based on the type of this information
         * about the output space.
         *
         * @param factory                             A reference to an object of type `IBinaryPredictorFactory` that
         *                                            should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CsrView` that provides row-wise
         *                                            access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IBinaryPredictor` that has
         *                                            been created
         */
        virtual std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix, const RuleList& model,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `ISparseBinaryPredictor`, based on the type of this
         * information about the output space.
         *
         * @param factory                             A reference to an object of type `ISparseBinaryPredictorFactory`
         *                                            that should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CContiguousView` that provides
         *                                            row-wise access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `ISparseBinaryPredictor`
         *                                            that has been created
         */
        virtual std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `ISparseBinaryPredictor`, based on the type of this
         * information about the output space.
         *
         * @param factory                             A reference to an object of type `ISparseBinaryPredictorFactory`
         *                                            that should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CsrView` that provides row-wise
         *                                            access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `ISparseBinaryPredictor`
         *                                            that has been created
         */
        virtual std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IScorePredictor`, based on the type of this information
         * about the output space.
         *
         * @param factory       A reference to an object of type `IScorePredictorFactory` that should be used to create
         *                      the instance
         * @param featureMatrix A reference to an object of type `CContiguousView` that provides row-wise access to the
         *                      features of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numOutputs    The number of outputs to predict for
         * @return              An unique pointer to an object of type `IScorePredictor` that has been created
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(
          const IScorePredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const RuleList& model, uint32 numOutputs) const = 0;

        /**
         * Creates and returns a new instance of the class `IScorePredictor`, based on the type of this information
         * about the output space.
         *
         * @param factory       A reference to an object of type `IScorePredictorFactory` that should be used to create
         *                      the instance
         * @param featureMatrix A reference to an object of type `CsrView` that provides row-wise access to the features
         *                      of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numOutputs    The number of outputs to predict for
         * @return              An unique pointer to an object of type `IScorePredictor` that has been created
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                                      const CsrView<const float32>& featureMatrix,
                                                                      const RuleList& model,
                                                                      uint32 numOutputs) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this
         * information about the output space.
         *
         * @param factory                             A reference to an object of type `IProbabilityPredictorFactory`
         *                                            that should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CContiguousView` that provides
         *                                            row-wise access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IProbabilityPredictor`
         *                                            that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this
         * information about the output space.
         *
         * @param factory                             A reference to an object of type `IProbabilityPredictorFactory`
         *                                            that should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CsrView` that provides row-wise
         *                                            access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IProbabilityPredictor`
         *                                            that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;
};
