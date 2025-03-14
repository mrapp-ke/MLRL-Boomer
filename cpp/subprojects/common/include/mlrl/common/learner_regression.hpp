/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/example_weights.hpp"
#include "mlrl/common/input/feature_info.hpp"
#include "mlrl/common/input/feature_matrix_column_wise.hpp"
#include "mlrl/common/input/feature_matrix_row_wise.hpp"
#include "mlrl/common/input/regression_matrix_row_wise.hpp"
#include "mlrl/common/learner.hpp"

#include <memory>
#include <utility>

/**
 * Defines an interface for all rule learners that can be applied to regression problems.
 */
class MLRLCOMMON_API IRegressionRuleLearner {
    public:

        virtual ~IRegressionRuleLearner() {}

        /**
         * Applies the rule learner to given training examples and corresponding ground truth regression scores.
         *
         * @param exampleWeights    A reference to an object of type `IExampleWeights` that provides access to the
         *                          weights of the training examples
         * @param featureInfo       A reference to an object of type `IFeatureInfo` that provides information about the
         *                          types of individual features
         * @param featureMatrix     A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                          column-wise access to the feature values of the training examples
         * @param regressionMatrix  A reference to an object of type `IRowWiseRegressionMatrix` that provides row-wise
         *                          access to the ground truth regression scores of the training examples
         * @return                  An unique pointer to an object of type `ITrainingResult` that provides access to the
         *                          results of fitting the rule learner to the training data
         */
        virtual std::unique_ptr<ITrainingResult> fit(const IExampleWeights& exampleWeights,
                                                     const IFeatureInfo& featureInfo,
                                                     const IColumnWiseFeatureMatrix& featureMatrix,
                                                     const IRowWiseRegressionMatrix& regressionMatrix) const = 0;

        /**
         * Returns whether the rule learner is able to predict scores or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param trainingResult    A reference to an object of type `ITrainingResult` that provides access to the model
         *                          and additional information that should be used to obtain predictions
         * @return                  True, if the rule learner is able to predict scores, false otherwise
         */
        virtual bool canPredictScores(const IRowWiseFeatureMatrix& featureMatrix,
                                      const ITrainingResult& trainingResult) const {
            return this->canPredictScores(featureMatrix, trainingResult.getNumOutputs());
        }

        /**
         * Returns whether the rule learner is able to predict scores or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param numOutputs        The number of outputs to predict for
         * @return                  True, if the rule learner is able to predict scores, false otherwise
         */
        virtual bool canPredictScores(const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict scores for given query examples. If the
         * prediction of scores is not supported by the rule learner, a `std::runtime_error` is thrown.
         *
         * @throws std::runtime_exception   The exception that is thrown if the prediction of scores is not supported by
         *                                  the rule learner
         * @param featureMatrix             A reference to an object of type `IRowWiseFeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param trainingResult            A reference to an object of type `ITrainingResult` that provides access to
         *                                  the model and additional information that should be used to obtain
         *                                  predictions
         * @return                          An unique pointer to an object of type `IScorePredictor` that may be used to
         *                                  predict scores for the given query examples
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                      const ITrainingResult& trainingResult) const {
            return this->createScorePredictor(featureMatrix, *trainingResult.getRuleModel(),
                                              *trainingResult.getOutputSpaceInfo(), trainingResult.getNumOutputs());
        }

        /**
         * Creates and returns a predictor that may be used to predict scores for given query examples. If the
         * prediction of scores is not supported by the rule learner, a `std::runtime_error` is thrown.
         *
         * @throws std::runtime_exception The exception that is thrown if the prediction of scores is not supported by
         *                                the rule learner
         * @param featureMatrix           A reference to an object of type `IRowWiseFeatureMatrix` that provides
         *                                row-wise access to the feature values of the query examples
         * @param ruleModel               A reference to an object of type `IRuleModel` that should be used to obtain
         *                                predictions
         * @param outputSpaceInfo         A reference to an object of type `IOutputSpaceInfo` that provides information
         *                                about the output space that may be used as a basis for obtaining predictions
         * @param numOutputs              The number of outputs to predict for
         * @return                        An unique pointer to an object of type `IScorePredictor` that may be used to
         *                                predict scores for the given query examples
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                      const IRuleModel& ruleModel,
                                                                      const IOutputSpaceInfo& outputSpaceInfo,
                                                                      uint32 numOutputs) const = 0;
};
