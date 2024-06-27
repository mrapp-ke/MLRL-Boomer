/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/binning/label_binning_auto.hpp"
#include "mlrl/boosting/binning/label_binning_equal_width.hpp"
#include "mlrl/boosting/learner.hpp"
#include "mlrl/boosting/losses/loss_decomposable_logistic.hpp"
#include "mlrl/boosting/losses/loss_decomposable_squared_hinge.hpp"
#include "mlrl/boosting/losses/loss_non_decomposable_logistic.hpp"
#include "mlrl/boosting/losses/loss_non_decomposable_squared_hinge.hpp"
#include "mlrl/boosting/prediction/predictor_binary_auto.hpp"
#include "mlrl/boosting/prediction/predictor_binary_example_wise.hpp"
#include "mlrl/boosting/prediction/predictor_binary_gfm.hpp"
#include "mlrl/boosting/prediction/predictor_binary_output_wise.hpp"
#include "mlrl/boosting/prediction/predictor_probability_auto.hpp"
#include "mlrl/boosting/prediction/predictor_probability_marginalized.hpp"
#include "mlrl/boosting/prediction/predictor_probability_output_wise.hpp"
#include "mlrl/boosting/prediction/probability_calibration_isotonic.hpp"
#include "mlrl/boosting/rule_model_assemblage/default_rule_auto.hpp"
#include "mlrl/boosting/sampling/partition_sampling_auto.hpp"
#include "mlrl/boosting/statistics/statistic_format_auto.hpp"
#include "mlrl/boosting/statistics/statistic_format_sparse.hpp"
#include "mlrl/common/learner_classification.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * Defines an interface for all rule learners that make use of gradient boosting for solving classification
     * problems.
     */
    class MLRLBOOSTING_API IBoostedClassificationRuleLearner : virtual public IBoostedRuleLearner,
                                                               virtual public IClassificationRuleLearner {
        public:

            virtual ~IBoostedClassificationRuleLearner() override {}

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide
             * whether a holdout set should be used or not.
             */
            class IAutomaticPartitionSamplingMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticPartitionSamplingMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide whether a holdout set should be used or not.
                     */
                    virtual void useAutomaticPartitionSampling() {
                        std::unique_ptr<IPartitionSamplingConfig>& partitionSamplingConfigPtr =
                          this->getPartitionSamplingConfigPtr();
                        partitionSamplingConfigPtr = std::make_unique<AutomaticPartitionSamplingConfig>(
                          this->getGlobalPruningConfigPtr(), this->getMarginalProbabilityCalibratorConfigPtr(),
                          this->getJointProbabilityCalibratorConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to not induce a default rule.
             */
            class INoDefaultRuleMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~INoDefaultRuleMixin() override {}

                    /**
                     * Configures the rule learner to not induce a default rule.
                     */
                    virtual void useNoDefaultRule() {
                        std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr = this->getDefaultRuleConfigPtr();
                        defaultRuleConfigPtr = std::make_unique<DefaultRuleConfig>(false);
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide
             * whether a default rule should be induced or not.
             */
            class IAutomaticDefaultRuleMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticDefaultRuleMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide whether a default rule should be induced or
                     * not.
                     */
                    virtual void useAutomaticDefaultRule() {
                        std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr = this->getDefaultRuleConfigPtr();
                        defaultRuleConfigPtr = std::make_unique<AutomaticDefaultRuleConfig>(
                          this->getStatisticsConfigPtr(), this->getLossConfigPtr(), this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a sparse
             * representation of gradients and Hessians, if possible.
             */
            class ISparseStatisticsMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~ISparseStatisticsMixin() override {}

                    /**
                     * Configures the rule learner to use a sparse representation of gradients and Hessians, if
                     * possible.
                     */
                    virtual void useSparseStatistics() {
                        std::unique_ptr<IStatisticsConfig>& statisticsConfigPtr = this->getStatisticsConfigPtr();
                        statisticsConfigPtr = std::make_unique<SparseStatisticsConfig>(this->getLossConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide
             * whether a dense or sparse representation of gradients and Hessians should be used.
             */
            class IAutomaticStatisticsMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticStatisticsMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide whether a dense or sparse representation of
                     * gradients and Hessians should be used.
                     */
                    virtual void useAutomaticStatistics() {
                        std::unique_ptr<IStatisticsConfig>& statisticsConfigPtr = this->getStatisticsConfigPtr();
                        statisticsConfigPtr = std::make_unique<AutomaticStatisticsConfig>(
                          this->getLossConfigPtr(), this->getHeadConfigPtr(), this->getDefaultRuleConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multivariate variant of the logistic loss that is non-decomposable.
             */
            class INonDecomposableLogisticLossMixin : virtual public IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~INonDecomposableLogisticLossMixin() override {}

                    /**
                     * Configures the rule learner to use a loss function that implements a multivariate variant of the
                     * logistic loss that is non-decomposable.
                     */
                    virtual void useNonDecomposableLogisticLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<NonDecomposableLogisticLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multivariate variant of the squared hinge loss that is non-decomposable.
             */
            class INonDecomposableSquaredHingeLossMixin : virtual public IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~INonDecomposableSquaredHingeLossMixin() override {}

                    /**
                     * Configures the rule learner to use a loss function that implements a multivariate variant of the
                     * squared hinge loss that is non-decomposable.
                     */
                    virtual void useNonDecomposableSquaredHingeLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr =
                          std::make_unique<NonDecomposableSquaredHingeLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multivariate variant of the logistic loss that is decomposable.
             */
            class IDecomposableLogisticLossMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IDecomposableLogisticLossMixin() override {}

                    /**
                     * Configures the rule learner to use a loss function that implements a multivariate variant of the
                     * logistic loss that is applied decomposable.
                     */
                    virtual void useDecomposableLogisticLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<DecomposableLogisticLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multivariate variant of the squared hinge loss that is decomposable.
             */
            class IDecomposableSquaredHingeLossMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IDecomposableSquaredHingeLossMixin() override {}

                    /**
                     * Configures the rule learner to use a loss function that implements a multivariate variant of the
                     * squared hinge loss that is decomposable.
                     */
                    virtual void useDecomposableSquaredHingeLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<DecomposableSquaredHingeLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a method for the
             * assignment of labels to bins.
             */
            class IEqualWidthLabelBinningMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IEqualWidthLabelBinningMixin() override {}

                    /**
                     * Configures the rule learner to use a method for the assignment of labels to bins in a way such
                     * that each bin contains labels for which the predicted score is expected to belong to the same
                     * value range.
                     *
                     * @return A reference to an object of type `IEqualWidthLabelBinningConfig` that allows further
                     *         configuration of the method for the assignment of labels to bins
                     */
                    virtual IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning() {
                        std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr = this->getLabelBinningConfigPtr();
                        std::unique_ptr<EqualWidthLabelBinningConfig> ptr =
                          std::make_unique<EqualWidthLabelBinningConfig>(this->getL1RegularizationConfigPtr(),
                                                                         this->getL2RegularizationConfigPtr());
                        IEqualWidthLabelBinningConfig& ref = *ptr;
                        labelBinningConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide
             * whether a method for the assignment of labels to bins should be used or not.
             */
            class IAutomaticLabelBinningMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticLabelBinningMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide whether a method for the assignment of labels
                     * to bins should be used or not.
                     */
                    virtual void useAutomaticLabelBinning() {
                        std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr = this->getLabelBinningConfigPtr();
                        labelBinningConfigPtr = std::make_unique<AutomaticLabelBinningConfig>(
                          this->getL1RegularizationConfigPtr(), this->getL2RegularizationConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to calibrate marginal
             * probabilities via isotonic regression.
             *
             */
            class IIsotonicMarginalProbabilityCalibrationMixin
                : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IIsotonicMarginalProbabilityCalibrationMixin() override {}

                    /**
                     * Configures the rule learner to calibrate marginal probabilities via isotonic regression.
                     *
                     * @return A reference to an object of type `IIsotonicMarginalProbabilityCalibratorConfig` that
                     *         allows further configuration of the calibrator
                     */
                    virtual IIsotonicMarginalProbabilityCalibratorConfig& useIsotonicMarginalProbabilityCalibration() {
                        std::unique_ptr<IMarginalProbabilityCalibratorConfig>& marginalProbabilityCalibratorConfigPtr =
                          this->getMarginalProbabilityCalibratorConfigPtr();
                        std::unique_ptr<IsotonicMarginalProbabilityCalibratorConfig> ptr =
                          std::make_unique<IsotonicMarginalProbabilityCalibratorConfig>(this->getLossConfigPtr());
                        IIsotonicMarginalProbabilityCalibratorConfig& ref = *ptr;
                        marginalProbabilityCalibratorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to calibrate joint
             * probabilities via isotonic regression.
             */
            class IIsotonicJointProbabilityCalibrationMixin
                : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IIsotonicJointProbabilityCalibrationMixin() override {}

                    /**
                     * Configures the rule learner to calibrate joint probabilities via isotonic regression.
                     *
                     * @return A reference to an object of type `IIsotonicJointProbabilityCalibratorConfig` that allows
                     *         further configuration of the calibrator
                     */
                    virtual IIsotonicJointProbabilityCalibratorConfig& useIsotonicJointProbabilityCalibration() {
                        std::unique_ptr<IJointProbabilityCalibratorConfig>& jointProbabilityCalibratorConfigPtr =
                          this->getJointProbabilityCalibratorConfigPtr();
                        std::unique_ptr<IsotonicJointProbabilityCalibratorConfig> ptr =
                          std::make_unique<IsotonicJointProbabilityCalibratorConfig>(this->getLossConfigPtr());
                        IIsotonicJointProbabilityCalibratorConfig& ref = *ptr;
                        jointProbabilityCalibratorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor that
             * predicts label-wise probabilities for given query examples by transforming the individual scores that are
             * predicted for each label into probabilities.
             */
            class IOutputWiseProbabilityPredictorMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IOutputWiseProbabilityPredictorMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor that predicts label-wise probabilities for given
                     * query examples by transforming the individual scores that are predicted for each label into
                     * probabilities.
                     *
                     * @return A reference to an object of type `IOutputWiseProbabilityPredictorConfig` that allows
                     *         further configuration of the predictor
                     */
                    virtual IOutputWiseProbabilityPredictorConfig& useOutputWiseProbabilityPredictor() {
                        std::unique_ptr<IProbabilityPredictorConfig>& probabilityPredictorConfigPtr =
                          this->getProbabilityPredictorConfigPtr();
                        std::unique_ptr<OutputWiseProbabilityPredictorConfig> ptr =
                          std::make_unique<OutputWiseProbabilityPredictorConfig>(
                            this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                        IOutputWiseProbabilityPredictorConfig& ref = *ptr;
                        probabilityPredictorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use predictor that
             * predicts label-wise probabilities for given query examples by marginalizing over the joint probabilities
             * of known label vectors.
             */
            class IMarginalizedProbabilityPredictorMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IMarginalizedProbabilityPredictorMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor that predicts label-wise probabilities for given
                     * query examples by marginalizing over the joint probabilities of known label vectors.
                     *
                     * @return A reference to an object of type `IMarginalizedProbabilityPredictorConfig` that allows
                     *         further configuration of the predictor
                     */
                    virtual IMarginalizedProbabilityPredictorConfig& useMarginalizedProbabilityPredictor() {
                        std::unique_ptr<IProbabilityPredictorConfig>& probabilityPredictorConfigPtr =
                          this->getProbabilityPredictorConfigPtr();
                        std::unique_ptr<MarginalizedProbabilityPredictorConfig> ptr =
                          std::make_unique<MarginalizedProbabilityPredictorConfig>(
                            this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                        IMarginalizedProbabilityPredictorConfig& ref = *ptr;
                        probabilityPredictorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide for a
             * predictor for predicting probability estimates.
             */
            class IAutomaticProbabilityPredictorMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticProbabilityPredictorMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide for a predictor for predicting probability
                     * estimates.
                     */
                    virtual void useAutomaticProbabilityPredictor() {
                        std::unique_ptr<IProbabilityPredictorConfig>& probabilityPredictorConfigPtr =
                          this->getProbabilityPredictorConfigPtr();
                        probabilityPredictorConfigPtr = std::make_unique<AutomaticProbabilityPredictorConfig>(
                          this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor that
             * predicts whether individual labels of given query examples are relevant or irrelevant by discretizing the
             * individual scores or probability estimates that are predicted for each label.
             */
            class IOutputWiseBinaryPredictorMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IOutputWiseBinaryPredictorMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor that predicts whether individual labels of given
                     * query examples are relevant or irrelevant by discretizing the individual scores or probability
                     * estimates that are predicted for each label.
                     *
                     * @return A reference to an object of type `IOutputWiseBinaryPredictorConfig` that allows further
                     *         configuration of the predictor
                     */
                    virtual IOutputWiseBinaryPredictorConfig& useOutputWiseBinaryPredictor() {
                        std::unique_ptr<IBinaryPredictorConfig>& binaryPredictorConfigPtr =
                          this->getBinaryPredictorConfigPtr();
                        std::unique_ptr<OutputWiseBinaryPredictorConfig> ptr =
                          std::make_unique<OutputWiseBinaryPredictorConfig>(this->getLossConfigPtr(),
                                                                            this->getParallelPredictionConfigPtr());
                        IOutputWiseBinaryPredictorConfig& ref = *ptr;
                        binaryPredictorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor that
             * predicts known label vectors for given query examples by comparing the predicted scores or probability
             * estimates to the label vectors encountered in the training data.
             */
            class IExampleWiseBinaryPredictorMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IExampleWiseBinaryPredictorMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor that predicts known label vectors for given query
                     * examples by comparing the predicted scores or probability estimates to the label vectors
                     * encountered in the training data.
                     *
                     * @return A reference to an object of type `IExampleWiseBinaryPredictorConfig` that allows further
                     *         configuration of the predictor
                     */
                    virtual IExampleWiseBinaryPredictorConfig& useExampleWiseBinaryPredictor() {
                        std::unique_ptr<IBinaryPredictorConfig>& binaryPredictorConfigPtr =
                          this->getBinaryPredictorConfigPtr();
                        std::unique_ptr<ExampleWiseBinaryPredictorConfig> ptr =
                          std::make_unique<ExampleWiseBinaryPredictorConfig>(this->getLossConfigPtr(),
                                                                             this->getParallelPredictionConfigPtr());
                        IExampleWiseBinaryPredictorConfig& ref = *ptr;
                        binaryPredictorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor that
             * predicts whether individual labels of given query examples are relevant or irrelevant by discretizing the
             * scores or probability estimates that are predicted for each label according to the general F-measure
             * maximizer (GFM).
             */
            class IGfmBinaryPredictorMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IGfmBinaryPredictorMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor that predicts whether individual labels of given
                     * query examples are relevant or irrelevant by discretizing the scores or probability estimates
                     * that are predicted for each label according to the general F-measure maximizer (GFM).
                     *
                     * @return A reference to an object of type `IGfmBinaryPredictorConfig` that allows further
                     *         configuration of the predictor
                     */
                    virtual IGfmBinaryPredictorConfig& useGfmBinaryPredictor() {
                        std::unique_ptr<IBinaryPredictorConfig>& binaryPredictorConfigPtr =
                          this->getBinaryPredictorConfigPtr();
                        std::unique_ptr<GfmBinaryPredictorConfig> ptr = std::make_unique<GfmBinaryPredictorConfig>(
                          this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                        IGfmBinaryPredictorConfig& ref = *ptr;
                        binaryPredictorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide for a
             * predictor for predicting whether individual labels are relevant or irrelevant.
             */
            class IAutomaticBinaryPredictorMixin : public virtual IBoostedClassificationRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticBinaryPredictorMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide for a predictor for predicting whether
                     * individual labels are relevant or irrelevant.
                     */
                    virtual void useAutomaticBinaryPredictor() {
                        std::unique_ptr<IBinaryPredictorConfig>& binaryPredictorConfigPtr =
                          this->getBinaryPredictorConfigPtr();
                        binaryPredictorConfigPtr = std::make_unique<AutomaticBinaryPredictorConfig>(
                          this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                    }
            };
    };
}
