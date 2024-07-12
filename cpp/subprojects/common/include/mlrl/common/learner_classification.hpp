/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_info.hpp"
#include "mlrl/common/input/feature_matrix_column_wise.hpp"
#include "mlrl/common/input/feature_matrix_row_wise.hpp"
#include "mlrl/common/input/label_matrix_row_wise.hpp"
#include "mlrl/common/learner.hpp"
#include "mlrl/common/sampling/instance_sampling_stratified_example_wise.hpp"
#include "mlrl/common/sampling/instance_sampling_stratified_output_wise.hpp"
#include "mlrl/common/sampling/partition_sampling_bi_stratified_example_wise.hpp"
#include "mlrl/common/sampling/partition_sampling_bi_stratified_output_wise.hpp"

#include <memory>
#include <utility>

/**
 * Defines an interface for all rule learners that can be applied to classification problems.
 */
class MLRLCOMMON_API IClassificationRuleLearner {
    public:

        virtual ~IClassificationRuleLearner() {}

        /**
         * Applies the rule learner to given training examples and corresponding ground truth labels.
         *
         * @param featureInfo       A reference to an object of type `IFeatureInfo` that provides information about the
         *                          types of individual features
         * @param featureMatrix     A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                          column-wise access to the feature values of the training examples
         * @param labelMatrix       A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access
         *                          to the ground truth labels of the training examples
         * @param randomState       The seed to be used by random number generators
         * @return                  An unique pointer to an object of type `ITrainingResult` that provides access to the
         *                          results of fitting the rule learner to the training data
         */
        virtual std::unique_ptr<ITrainingResult> fit(const IFeatureInfo& featureInfo,
                                                     const IColumnWiseFeatureMatrix& featureMatrix,
                                                     const IRowWiseLabelMatrix& labelMatrix,
                                                     uint32 randomState) const = 0;

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
                                      const ITrainingResult& trainingResult) const = 0;

        /**
         * Returns whether the rule learner is able to predict scores or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param numLabels         The number of labels to predict for
         * @return                  True, if the rule learner is able to predict scores, false otherwise
         */
        virtual bool canPredictScores(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

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
                                                                      const ITrainingResult& trainingResult) const = 0;

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
         * @param numLabels               The number of labels to predict for
         * @return                        An unique pointer to an object of type `IScorePredictor` that may be used to
         *                                predict scores for the given query examples
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                      const IRuleModel& ruleModel,
                                                                      const IOutputSpaceInfo& outputSpaceInfo,
                                                                      uint32 numLabels) const = 0;

        /**
         * Returns whether the rule learner is able to predict probabilities or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param trainingResult    A reference to an object of type `ITrainingResult` that provides access to the model
         *                          and additional information that should be used to obtain predictions
         * @return                  True, if the rule learner is able to predict probabilities, false otherwise
         */
        virtual bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix,
                                             const ITrainingResult& trainingResult) const = 0;

        /**
         * Returns whether the rule learner is able to predict probabilities or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param numLabels         The number of labels to predict for
         * @return                  True, if the rule learner is able to predict probabilities, false otherwise
         */
        virtual bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict probability estimates for given query examples.
         * If the prediction of probability estimates is not supported by the rule learner, a `std::runtime_error` is
         * thrown.
         *
         * @throws std::runtime_exception   The exception that is thrown if the prediction of probability estimates is
         *                                  not supported by the rule learner
         * @param featureMatrix             A reference to an object of type `IRowWiseFeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param trainingResult            A reference to an object of type `ITrainingResult` that provides access to
         *                                  the model and additional information that should be used to obtain
         *                                  predictions
         * @return                          An unique pointer to an object of type `IProbabilityPredictor` that may be
         *                                  used to predict probability estimates for the given query examples
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict probability estimates for given query examples.
         * If the prediction of probability estimates is not supported by the rule learner, a `std::runtime_error` is
         * thrown.
         *
         * @throws std::runtime_exception             The exception that is thrown if the prediction of probability
         *                                            estimates is not supported by the rule learner
         * @param featureMatrix                       A reference to an object of type `IRowWiseFeatureMatrix` that
         *                                            provides row-wise access to the feature values of the query
         *                                            examples
         * @param ruleModel                           A reference to an object of type `IRuleModel` that should be used
         *                                            to obtain predictions
         * @param outputSpaceInfo                     A reference to an object of type `IOutputSpaceInfo` that provides
         *                                            information about the output space that may be used as a basis for
         *                                            obtaining predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IProbabilityPredictor`
         *                                            that may be used to predict probability estimates for the given
         *                                            query examples
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Returns whether the rule learner is able to predict binary labels or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param trainingResult    A reference to an object of type `ITrainingResult` that provides access to the model
         *                          and additional information that should be used to obtain predictions
         * @return                  True, if the rule learner is able to predict binary labels, false otherwise
         */
        virtual bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix,
                                      const ITrainingResult& trainingResult) const = 0;

        /**
         * Returns whether the rule learner is able to predict binary labels or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param numLabels         The number of labels to predict for
         * @return                  True, if the rule learner is able to predict binary labels, false otherwise
         */
        virtual bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict binary labels for given query examples. If the
         * prediction of binary labels is not supported by the rule learner, a `std::runtime_error` is thrown.
         *
         * @throws std::runtime_exception   The exception that is thrown if the prediction of binary labels is not
         *                                  supported by the rule learner
         * @param featureMatrix             A reference to an object of type `IRowWiseFeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param trainingResult            A reference to an object of type `ITrainingResult` that provides access to
         *                                  the model and additional information that should be used to obtain
         *                                  predictions
         * @return                          An unique pointer to an object of type `IBinaryPredictor` that may be used
         *                                  to predict binary labels for the given query examples
         */
        virtual std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict binary labels for given query examples. If the
         * prediction of binary labels is not supported by the rule learner, a `std::runtime_error` is thrown.
         *
         * @throws std::runtime_exception             The exception that is thrown if the prediction of binary labels is
         *                                            not supported by the rule learner
         * @param featureMatrix                       A reference to an object of type `IRowWiseFeatureMatrix` that
         *                                            provides row-wise access to the feature values of the query
         *                                            examples
         * @param ruleModel                           A reference to an object of type `IRuleModel` that should be used
         *                                            to obtain predictions
         * @param outputSpaceInfo                     A reference to an object of type `IOutputSpaceInfo` that provides
         *                                            information about the output space that may be used as a basis for
         *                                            obtaining predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IBinaryPredictor` that may
         *                                            be used to predict binary labels for the given query examples
         */
        virtual std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict sparse binary labels for given query examples. If
         * the prediction of sparse binary labels is not supported by the rule learner, a `std::runtime_error` is
         * thrown.
         *
         * @throws std::runtime_exception   The exception that is thrown if the prediction of sparse binary labels is
         *                                  not supported by the rule learner
         * @param featureMatrix             A reference to an object of type `IRowWiseFeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param trainingResult            A reference to an object of type `ITrainingResult` that provides access to
         *                                  the model and additional information that should be used to obtain
         *                                  predictions
         * @return                          An unique pointer to an object of type `ISparseBinaryPredictor` that may be
         *                                  used to predict sparse binary labels for the given query examples
         */
        virtual std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict sparse binary labels for given query examples. If
         * the prediction of sparse binary labels is not supported by the rule learner, a `std::runtime_error` is
         * thrown.
         *
         * @throws std::runtime_exception             The exception that is thrown if the prediction of sparse binary
         *                                            labels is not supported by the rule learner
         * @param featureMatrix                       A reference to an object of type `IRowWiseFeatureMatrix` that
         *                                            provides row-wise access to the feature values of the query
         *                                            examples
         * @param ruleModel                           A reference to an object of type `IRuleModel` that should be used
         *                                            to obtain predictions
         * @param outputSpaceInfo                     A reference to an object of type `IOutputSpaceInfo` that provides
         *                                            information about the output space that may be used as a basis for
         *                                            obtaining predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `ISparseBinaryPredictor`
         *                                            that may be used to predict sparse binary labels for the given
         *                                            query examples
         */
        virtual std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use label-wise stratified instance
 * sampling.
 */
class MLRLCOMMON_API IOutputWiseStratifiedInstanceSamplingMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IOutputWiseStratifiedInstanceSamplingMixin() override {}

        /**
         * Configures the rule learner to sample from the available training examples using stratification, such that
         * for each label the proportion of relevant and irrelevant examples is maintained, whenever a new rule should
         * be learned.
         *
         * @return A reference to an object of type `IOutputWiseStratifiedInstanceSamplingConfig` that allows further
         *         configuration of the method for sampling instances
         */
        virtual IOutputWiseStratifiedInstanceSamplingConfig& useOutputWiseStratifiedInstanceSampling() {
            Property<IInstanceSamplingConfig> property = this->getInstanceSamplingConfig();
            std::unique_ptr<OutputWiseStratifiedInstanceSamplingConfig> ptr =
              std::make_unique<OutputWiseStratifiedInstanceSamplingConfig>();
            IOutputWiseStratifiedInstanceSamplingConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use example-wise stratified instance
 * sampling.
 */
class MLRLCOMMON_API IExampleWiseStratifiedInstanceSamplingMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IExampleWiseStratifiedInstanceSamplingMixin() override {}

        /**
         * Configures the rule learner to sample from the available training examples using stratification, where
         * distinct label vectors are treated as individual classes, whenever a new rule should be learned.
         *
         * @return A reference to an object of type `IExampleWiseStratifiedInstanceSamplingConfig` that allows further
         *         configuration of the method for sampling instances
         */
        virtual IExampleWiseStratifiedInstanceSamplingConfig& useExampleWiseStratifiedInstanceSampling() {
            Property<IInstanceSamplingConfig> property = this->getInstanceSamplingConfig();
            std::unique_ptr<ExampleWiseStratifiedInstanceSamplingConfig> ptr =
              std::make_unique<ExampleWiseStratifiedInstanceSamplingConfig>();
            IExampleWiseStratifiedInstanceSamplingConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to partition the available training
 * examples into a training set and a holdout set using stratification, such that for each label the proportion of
 * relevant and irrelevant examples is maintained.
 */
class MLRLCOMMON_API IOutputWiseStratifiedBiPartitionSamplingMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IOutputWiseStratifiedBiPartitionSamplingMixin() override {}

        /**
         * Configures the rule learner to partition the available training examples into a training set and a holdout
         * set using stratification, such that for each label the proportion of relevant and irrelevant examples is
         * maintained.
         *
         * @return A reference to an object of type `IOutputWiseStratifiedBiPartitionSamplingConfig` that allows further
         *         configuration of the method for partitioning the available training examples into a training and a
         *         holdout set
         */
        virtual IOutputWiseStratifiedBiPartitionSamplingConfig& useOutputWiseStratifiedBiPartitionSampling() {
            Property<IPartitionSamplingConfig> property = this->getPartitionSamplingConfig();
            std::unique_ptr<OutputWiseStratifiedBiPartitionSamplingConfig> ptr =
              std::make_unique<OutputWiseStratifiedBiPartitionSamplingConfig>();
            IOutputWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to partition the available training
 * examples into a training set and a holdout set using stratification, where distinct label vectors are treated as
 * individual classes.
 */
class MLRLCOMMON_API IExampleWiseStratifiedBiPartitionSamplingMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IExampleWiseStratifiedBiPartitionSamplingMixin() override {}

        /**
         * Configures the rule learner to partition the available training examples into a training set and a holdout
         * set using stratification, where distinct label vectors are treated as individual classes.
         *
         * @return A reference to an object of type `IExampleWiseStratifiedBiPartitionSamplingConfig` that allows
         *         further configuration of the method for partitioning the available training examples into a training
         *         and a holdout set
         */
        virtual IExampleWiseStratifiedBiPartitionSamplingConfig& useExampleWiseStratifiedBiPartitionSampling() {
            Property<IPartitionSamplingConfig> property = this->getPartitionSamplingConfig();
            std::unique_ptr<ExampleWiseStratifiedBiPartitionSamplingConfig> ptr =
              std::make_unique<ExampleWiseStratifiedBiPartitionSamplingConfig>();
            IExampleWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};
