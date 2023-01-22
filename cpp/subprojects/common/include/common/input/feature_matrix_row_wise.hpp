/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix.hpp"
#include <memory>

// Forward declarations
class IRuleModel;
class ILabelSpaceInfo;
class ILabelPredictor;
class ILabelPredictorFactory;
class ISparseLabelPredictor;
class ISparseLabelPredictorFactory;
class IScorePredictor;
class IScorePredictorFactory;
class IProbabilityPredictor;
class IProbabilityPredictorFactory;
template<typename T> class DensePredictionMatrix; // TODO Remove
class BinarySparsePredictionMatrix; // TODO Remove
class IClassificationPredictor; // TODO Remove
class IOldRegressionPredictor; // TODO Remove
class IOldProbabilityPredictor; // TODO Remove


/**
 * Defines an interface for all feature matrices that provide row-wise access to the feature values of examples.
 */
class MLRLCOMMON_API IRowWiseFeatureMatrix : virtual public IFeatureMatrix {

    public:

        virtual ~IRowWiseFeatureMatrix() override { };

        /**
         * Creates and returns a new instance of the class `ILabelPredictor`, based on the type of this feature matrix.
         *
         * @param factory           A reference to an object of type `ILabelPredictorFactory` that should be used to
         *                          create the instance
         * @param ruleModel         A reference to an object of type `IRuleModel` that should be used to obtain
         *                          predictions
         * @param labelSpaceInfo    A reference to an object of type `ILabelSpaceInfo` that provides information about
         *                          the label space that may be used as a basis for making predictions
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `ILabelPredictor` that has been created
         */
        virtual std::unique_ptr<ILabelPredictor> createLabelPredictor(const ILabelPredictorFactory& factory,
                                                                      const IRuleModel& ruleModel,
                                                                      const ILabelSpaceInfo& labelSpaceInfo,
                                                                      uint32 numLabels) const = 0;

        /**
         * Obtains and returns dense predictions for all examples in this feature matrix, using a specific
         * `IClassificationPredictor`, depending on the type of this feature matrix.
         *
         * @param predictor A reference to an object of type `IClassificationPredictor` that should be used to obtain
         *                  predictions
         * @param numLabels The number of labels to predict for
         * @return          An unique pointer to an object of type `DensePredictionMatrix` that stores the predictions
         */
        // TODO Remove
        virtual std::unique_ptr<DensePredictionMatrix<uint8>> predictLabels(
            const IClassificationPredictor& predictor, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `ISparseLabelPredictor`, based on the type of this feature
         * matrix.
         *
         * @param factory           A reference to an object of type `ISparseLabelPredictorFactory` that should be used
         *                          to create the instance
         * @param ruleModel         A reference to an object of type `IRuleModel` that should be used to obtain
         *                          predictions
         * @param labelSpaceInfo    A reference to an object of type `ILabelSpaceInfo` that provides information about
         *                          the label space that may be used as a basis for making predictions
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `ISparseLabelPredictor` that has been created
         */
        virtual std::unique_ptr<ISparseLabelPredictor> createSparseLabelPredictor(
            const ISparseLabelPredictorFactory& factory, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const = 0;

        /**
         * Obtains and returns sparse predictions for all examples in this feature matrix, using a specific
         * `IClassificationPredictor`, depending on the type of this feature matrix.
         *
         * @param predictor A reference to an object of type `IClassificationPredictor` that should be used to obtain
         *                  predictions
         * @param numLabels The number of labels to predict for
         * @return          An unique pointer to an object of type `BinarySparsePredictionMatrix` that stores the
         *                  predictions
         */
        // TODO Remove
        virtual std::unique_ptr<BinarySparsePredictionMatrix> predictSparseLabels(
            const IClassificationPredictor& predictor, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IScorePredictor`, based on the type of this feature matrix.
         *
         * @param factory           A reference to an object of type `IScorePredictorFactory` that should be used to
         *                          create the instance
         * @param ruleModel         A reference to an object of type `IRuleModel` that should be used to obtain
         *                          predictions
         * @param labelSpaceInfo    A reference to an object of type `ILabelSpaceInfo` that provides information about
         *                          the label space that may be used as a basis for making predictions
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `IScorePredictor` that has been created
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                                      const IRuleModel& ruleModel,
                                                                      const ILabelSpaceInfo& labelSpaceInfo,
                                                                      uint32 numLabels) const = 0;

        /**
         * Obtains and returns regression scores for all examples in this feature matrix, using a specific
         * `IRegressionPredictor`, depending on the type of this feature matrix.
         *
         * @param predictor A reference to an object of type `IRegressionPredictor` that should be used to obtain
         *                  predictions
         * @param numLabels The number of labels to predict for
         * @return          An unique pointer to an object of type `DensePredictionMatrix` that stores the predictions
         */
        // TODO Remove
        virtual std::unique_ptr<DensePredictionMatrix<float64>> predictScores(
            const IOldRegressionPredictor& predictor, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this feature
         * matrix.
         *
         * @param factory           A reference to an object of type `IProbabilityPredictorFactory` that should be used
         *                          to create the instance
         * @param ruleModel         A reference to an object of type `IRuleModel` that should be used to obtain
         *                          predictions
         * @param labelSpaceInfo    A reference to an object of type `ILabelSpaceInfo` that provides information about
         *                          the label space that may be used as a basis for making predictions
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `IProbabilityPredictor` that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
            const IProbabilityPredictorFactory& factory, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const = 0;

        /**
         * Obtains and returns probability estimates for all examples in this feature matrix, using a specific
         * `IProbabilityPredictor`, depending on the type of this feature matrix.
         *
         * @param predictor A reference to an object of type `IProbabilityPredictor` that should be used to obtain
         *                  predictions
         * @param numLabels The number of labels to predict for
         * @return          An unique pointer to an object of type `DensePredictionMatrix` that stores the predictions
         */
        // TODO Remove
        virtual std::unique_ptr<DensePredictionMatrix<float64>> predictProbabilities(
            const IOldProbabilityPredictor& predictor, uint32 numLabels) const = 0;

};
