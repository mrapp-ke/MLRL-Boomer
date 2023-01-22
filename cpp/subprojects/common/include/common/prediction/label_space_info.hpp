/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include "common/macros.hpp"
#include <memory>

class CContiguousFeatureMatrix;
class CsrFeatureMatrix;
class RuleList;
class ILabelPredictor;
class ILabelPredictorFactory;
class ISparseLabelPredictor;
class ISparseLabelPredictorFactory;
class IScorePredictor;
class IScorePredictorFactory;
class IProbabilityPredictor;
class IProbabilityPredictorFactory;
class IClassificationPredictorFactory; // TODO Remove
class IClassificationPredictor; // TODO Remove
class IRegressionPredictorFactory; // TODO Remove
class IOldRegressionPredictor; // TODO Remove
class IOldProbabilityPredictorFactory; // TODO Remove
class IOldProbabilityPredictor; // TODO Remove


/**
 * Defines an interface for all classes that provide information about the label space that may be used as a basis for
 * making predictions.
 */
class MLRLCOMMON_API ILabelSpaceInfo {

    public:

        virtual ~ILabelSpaceInfo() { };

        /**
         * Creates and returns a new instance of the class `ILabelPredictor`, based on the type of this information
         * about the label space.
         *
         * @param factory       A reference to an object of type `ILabelPredictorFactory` that should be used to create
         *                      the instance
         * @param featureMatrix A reference to an object of type `CContiguousFeatureMatrix` that provides row-wise
         *                      access to the features of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `ILabelPredictor` that has been created
         */
        virtual std::unique_ptr<ILabelPredictor> createLabelPredictor(const ILabelPredictorFactory& factory,
                                                                      const CContiguousFeatureMatrix& featureMatrix,
                                                                      const RuleList& model,
                                                                      uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `ILabelPredictor`, based on the type of this information
         * about the label space.
         *
         * @param factory       A reference to an object of type `ILabelPredictorFactory` that should be used to create
         *                      the instance
         * @param featureMatrix A reference to an object of type `CsrFeatureMatrix` that provides row-wise access to the
         *                      features of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `ILabelPredictor` that has been created
         */
        virtual std::unique_ptr<ILabelPredictor> createLabelPredictor(const ILabelPredictorFactory& factory,
                                                                      const CsrFeatureMatrix& featureMatrix,
                                                                      const RuleList& model,
                                                                      uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `ISparseLabelPredictor`, based on the type of this
         * information about the label space.
         *
         * @param factory       A reference to an object of type `ISparseLabelPredictorFactory` that should be used to
         *                      create the instance
         * @param featureMatrix A reference to an object of type `CContiguousFeatureMatrix` that provides row-wise
         *                      access to the features of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `ISparseLabelPredictor` that has been created
         */
        virtual std::unique_ptr<ISparseLabelPredictor> createSparseLabelPredictor(
            const ISparseLabelPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
            const RuleList& model, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `ISparseLabelPredictor`, based on the type of this
         * information about the label space.
         *
         * @param factory       A reference to an object of type `ISparseLabelPredictorFactory` that should be used to
         *                      create the instance
         * @param featureMatrix A reference to an object of type `CsrFeatureMatrix` that provides row-wise access to the
         *                      features of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `ISparseLabelPredictor` that has been created
         */
        virtual std::unique_ptr<ISparseLabelPredictor> createSparseLabelPredictor(
            const ISparseLabelPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
            uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IClassificationPredictor`, based on the type of this
         * information about the label space.
         *
         * @param factory   A reference to an object of type `IClassificationPredictorFactory` that should be used to
         *                  create the instance
         * @param model     A reference to an object of type `RuleList` that should be used to obtain predictions
         * @return          An unique pointer to an object of type `IClassificationPredictor` that has been created
         */
        // TODO Remove
        virtual std::unique_ptr<IClassificationPredictor> createClassificationPredictor(
            const IClassificationPredictorFactory& factory, const RuleList& model) const = 0;

        /**
         * Creates and returns a new instance of the class `IScorePredictor`, based on the type of this information
         * about the label space.
         *
         * @param factory       A reference to an object of type `IScorePredictorFactory` that should be used to create
         *                      the instance
         * @param featureMatrix A reference to an object of type `CContiguousFeatureMatrix` that provides row-wise
         *                      access to the features of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IScorePredictor` that has been created
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                                      const CContiguousFeatureMatrix& featureMatrix,
                                                                      const RuleList& model,
                                                                      uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IScorePredictor`, based on the type of this information
         * about the label space.
         *
         * @param factory       A reference to an object of type `IScorePredictorFactory` that should be used to create
         *                      the instance
         * @param featureMatrix A reference to an object of type `CsrFeatureMatrix` that provides row-wise access to the
         *                      features of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IScorePredictor` that has been created
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                                      const CsrFeatureMatrix& featureMatrix,
                                                                      const RuleList& model,
                                                                      uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IRegressionPredictor`, based on the type of this information
         * about the label space.
         *
         * @param factory   A reference to an object of type `IRegressionPredictorFactory` that should be used to create
         *                  the instance
         * @param model     A reference to an object of type `RuleList` that should be used to obtain predictions
         * @return          An unique pointer to an object of type `IRegressionPredictor` that has been created
         */
        // TODO Remove
        virtual std::unique_ptr<IOldRegressionPredictor> createRegressionPredictor(
            const IRegressionPredictorFactory& factory, const RuleList& model) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this
         * information about the label space.
         *
         * @param factory       A reference to an object of type `IProbabilityPredictorFactory` that should be used to
         *                      create the instance
         * @param featureMatrix A reference to an object of type `CContiguousFeatureMatrix` that provides row-wise
         *                      access to the features of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IProbabilityPredictor` that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
            const IProbabilityPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
            const RuleList& model, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this
         * information about the label space.
         *
         * @param factory       A reference to an object of type `IProbabilityPredictorFactory` that should be used to
         *                      create the instance
         * @param featureMatrix A reference to an object of type `CsrFeatureMatrix` that provides row-wise access to the
         *                      features of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IProbabilityPredictor` that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
            const IProbabilityPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
            uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this
         * information about the label space.
         *
         * @param factory   A reference to an object of type `IProbabilityPredictorFactory` that should be used to
         *                  create the instance
         * @param model     A reference to an object of type `RuleList` that should be used to obtain predictions
         * @return          An unique pointer to an object of type `IProbabilityPredictor` that has been created
         */
        // TODO Remove
        virtual std::unique_ptr<IOldProbabilityPredictor> createProbabilityPredictor(
            const IOldProbabilityPredictorFactory& factory, const RuleList& model) const = 0;

};
