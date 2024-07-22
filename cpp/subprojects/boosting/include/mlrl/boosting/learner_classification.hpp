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
#include "mlrl/common/learner_classification.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
     * implements a multivariate variant of the logistic loss that is non-decomposable.
     */
    class MLRLBOOSTING_API INonDecomposableLogisticLossMixin : virtual public IBoostedRuleLearnerConfig {
        public:

            virtual ~INonDecomposableLogisticLossMixin() override {}

            /**
             * Configures the rule learner to use a loss function that implements a multivariate variant of the logistic
             * loss that is non-decomposable.
             */
            virtual void useNonDecomposableLogisticLoss() {
                this->getClassificationLossConfig().set(
                  std::make_unique<NonDecomposableLogisticLossConfig>(this->getHeadConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
     * implements a multivariate variant of the squared hinge loss that is non-decomposable.
     */
    class MLRLBOOSTING_API INonDecomposableSquaredHingeLossMixin : virtual public IBoostedRuleLearnerConfig {
        public:

            virtual ~INonDecomposableSquaredHingeLossMixin() override {}

            /**
             * Configures the rule learner to use a loss function that implements a multivariate variant of the squared
             * hinge loss that is non-decomposable.
             */
            virtual void useNonDecomposableSquaredHingeLoss() {
                this->getClassificationLossConfig().set(
                  std::make_unique<NonDecomposableSquaredHingeLossConfig>(this->getHeadConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
     * implements a multivariate variant of the logistic loss that is decomposable.
     */
    class MLRLBOOSTING_API IDecomposableLogisticLossMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IDecomposableLogisticLossMixin() override {}

            /**
             * Configures the rule learner to use a loss function that implements a multivariate variant of the logistic
             * loss that is applied decomposable.
             */
            virtual void useDecomposableLogisticLoss() {
                this->getClassificationLossConfig().set(
                  std::make_unique<DecomposableLogisticLossConfig>(this->getHeadConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
     * implements a multivariate variant of the squared hinge loss that is decomposable.
     */
    class MLRLBOOSTING_API IDecomposableSquaredHingeLossMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IDecomposableSquaredHingeLossMixin() override {}

            /**
             * Configures the rule learner to use a loss function that implements a multivariate variant of the squared
             * hinge loss that is decomposable.
             */
            virtual void useDecomposableSquaredHingeLoss() {
                this->getClassificationLossConfig().set(
                  std::make_unique<DecomposableSquaredHingeLossConfig>(this->getHeadConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a method for the assignment of
     * labels to bins.
     */
    class MLRLBOOSTING_API IEqualWidthLabelBinningMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IEqualWidthLabelBinningMixin() override {}

            /**
             * Configures the rule learner to use a method for the assignment of labels to bins in a way such that each
             * bin contains labels for which the predicted score is expected to belong to the same value range.
             *
             * @return A reference to an object of type `IEqualWidthLabelBinningConfig` that allows further
             *         configuration of the method for the assignment of labels to bins
             */
            virtual IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning() {
                auto ptr = std::make_unique<EqualWidthLabelBinningConfig>(this->getL1RegularizationConfig(),
                                                                          this->getL2RegularizationConfig());
                IEqualWidthLabelBinningConfig& ref = *ptr;
                this->getLabelBinningConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to automatically decide whether a
     * method for the assignment of labels to bins should be used or not.
     */
    class MLRLBOOSTING_API IAutomaticLabelBinningMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IAutomaticLabelBinningMixin() override {}

            /**
             * Configures the rule learner to automatically decide whether a method for the assignment of labels to bins
             * should be used or not.
             */
            virtual void useAutomaticLabelBinning() {
                this->getLabelBinningConfig().set(std::make_unique<AutomaticLabelBinningConfig>(
                  this->getL1RegularizationConfig(), this->getL2RegularizationConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to calibrate marginal probabilities
     * via isotonic regression.
     *
     */
    class MLRLBOOSTING_API IIsotonicMarginalProbabilityCalibrationMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IIsotonicMarginalProbabilityCalibrationMixin() override {}

            /**
             * Configures the rule learner to calibrate marginal probabilities via isotonic regression.
             *
             * @return A reference to an object of type `IIsotonicMarginalProbabilityCalibratorConfig` that allows
             *         further configuration of the calibrator
             */
            virtual IIsotonicMarginalProbabilityCalibratorConfig& useIsotonicMarginalProbabilityCalibration() {
                auto ptr =
                  std::make_unique<IsotonicMarginalProbabilityCalibratorConfig>(this->getClassificationLossConfig());
                IIsotonicMarginalProbabilityCalibratorConfig& ref = *ptr;
                this->getMarginalProbabilityCalibratorConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to calibrate joint probabilities via
     * isotonic regression.
     */
    class MLRLBOOSTING_API IIsotonicJointProbabilityCalibrationMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IIsotonicJointProbabilityCalibrationMixin() override {}

            /**
             * Configures the rule learner to calibrate joint probabilities via isotonic regression.
             *
             * @return A reference to an object of type `IIsotonicJointProbabilityCalibratorConfig` that allows further
             *         configuration of the calibrator
             */
            virtual IIsotonicJointProbabilityCalibratorConfig& useIsotonicJointProbabilityCalibration() {
                auto ptr =
                  std::make_unique<IsotonicJointProbabilityCalibratorConfig>(this->getClassificationLossConfig());
                IIsotonicJointProbabilityCalibratorConfig& ref = *ptr;
                this->getJointProbabilityCalibratorConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a predictor that predicts
     * label-wise probabilities for given query examples by transforming the individual scores that are predicted for
     * each label into probabilities.
     */
    class MLRLBOOSTING_API IOutputWiseProbabilityPredictorMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IOutputWiseProbabilityPredictorMixin() override {}

            /**
             * Configures the rule learner to use a predictor that predicts label-wise probabilities for given query
             * examples by transforming the individual scores that are predicted for each label into probabilities.
             *
             * @return A reference to an object of type `IOutputWiseProbabilityPredictorConfig` that allows further
             *         configuration of the predictor
             */
            virtual IOutputWiseProbabilityPredictorConfig& useOutputWiseProbabilityPredictor() {
                auto ptr = std::make_unique<OutputWiseProbabilityPredictorConfig>(this->getClassificationLossConfig(),
                                                                                  this->getParallelPredictionConfig());
                IOutputWiseProbabilityPredictorConfig& ref = *ptr;
                this->getProbabilityPredictorConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use predictor that predicts
     * label-wise probabilities for given query examples by marginalizing over the joint probabilities of known label
     * vectors.
     */
    class MLRLBOOSTING_API IMarginalizedProbabilityPredictorMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IMarginalizedProbabilityPredictorMixin() override {}

            /**
             * Configures the rule learner to use a predictor that predicts label-wise probabilities for given query
             * examples by marginalizing over the joint probabilities of known label vectors.
             *
             * @return A reference to an object of type `IMarginalizedProbabilityPredictorConfig` that allows further
             *         configuration of the predictor
             */
            virtual IMarginalizedProbabilityPredictorConfig& useMarginalizedProbabilityPredictor() {
                auto ptr = std::make_unique<MarginalizedProbabilityPredictorConfig>(
                  this->getClassificationLossConfig(), this->getParallelPredictionConfig());
                IMarginalizedProbabilityPredictorConfig& ref = *ptr;
                this->getProbabilityPredictorConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to automatically decide for a
     * predictor for predicting probability estimates.
     */
    class MLRLBOOSTING_API IAutomaticProbabilityPredictorMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IAutomaticProbabilityPredictorMixin() override {}

            /**
             * Configures the rule learner to automatically decide for a predictor for predicting probability estimates.
             */
            virtual void useAutomaticProbabilityPredictor() {
                this->getProbabilityPredictorConfig().set(std::make_unique<AutomaticProbabilityPredictorConfig>(
                  this->getClassificationLossConfig(), this->getParallelPredictionConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a predictor that predicts
     * whether individual labels of given query examples are relevant or irrelevant by discretizing the individual
     * scores or probability estimates that are predicted for each label.
     */
    class MLRLBOOSTING_API IOutputWiseBinaryPredictorMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IOutputWiseBinaryPredictorMixin() override {}

            /**
             * Configures the rule learner to use a predictor that predicts whether individual labels of given query
             * examples are relevant or irrelevant by discretizing the individual scores or probability estimates that
             * are predicted for each label.
             *
             * @return A reference to an object of type `IOutputWiseBinaryPredictorConfig` that allows further
             *         configuration of the predictor
             */
            virtual IOutputWiseBinaryPredictorConfig& useOutputWiseBinaryPredictor() {
                auto ptr = std::make_unique<OutputWiseBinaryPredictorConfig>(this->getClassificationLossConfig(),
                                                                             this->getParallelPredictionConfig());
                IOutputWiseBinaryPredictorConfig& ref = *ptr;
                this->getBinaryPredictorConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a predictor that predicts
     * known label vectors for given query examples by comparing the predicted scores or probability estimates to the
     * label vectors encountered in the training data.
     */
    class MLRLBOOSTING_API IExampleWiseBinaryPredictorMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IExampleWiseBinaryPredictorMixin() override {}

            /**
             * Configures the rule learner to use a predictor that predicts known label vectors for given query examples
             * by comparing the predicted scores or probability estimates to the label vectors encountered in the
             * training data.
             *
             * @return A reference to an object of type `IExampleWiseBinaryPredictorConfig` that allows further
             *         configuration of the predictor
             */
            virtual IExampleWiseBinaryPredictorConfig& useExampleWiseBinaryPredictor() {
                auto ptr = std::make_unique<ExampleWiseBinaryPredictorConfig>(this->getClassificationLossConfig(),
                                                                              this->getParallelPredictionConfig());
                IExampleWiseBinaryPredictorConfig& ref = *ptr;
                this->getBinaryPredictorConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a predictor that predicts
     * whether individual labels of given query examples are relevant or irrelevant by discretizing the scores or
     * probability estimates that are predicted for each label according to the general F-measure maximizer (GFM).
     */
    class MLRLBOOSTING_API IGfmBinaryPredictorMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IGfmBinaryPredictorMixin() override {}

            /**
             * Configures the rule learner to use a predictor that predicts whether individual labels of given query
             * examples are relevant or irrelevant by discretizing the scores or probability estimates that are
             * predicted for each label according to the general F-measure maximizer (GFM).
             *
             * @return A reference to an object of type `IGfmBinaryPredictorConfig` that allows further configuration of
             *         the predictor
             */
            virtual IGfmBinaryPredictorConfig& useGfmBinaryPredictor() {
                auto ptr = std::make_unique<GfmBinaryPredictorConfig>(this->getClassificationLossConfig(),
                                                                      this->getParallelPredictionConfig());
                IGfmBinaryPredictorConfig& ref = *ptr;
                this->getBinaryPredictorConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to automatically decide for a
     * predictor for predicting whether individual labels are relevant or irrelevant.
     */
    class MLRLBOOSTING_API IAutomaticBinaryPredictorMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IAutomaticBinaryPredictorMixin() override {}

            /**
             * Configures the rule learner to automatically decide for a predictor for predicting whether individual
             * labels are relevant or irrelevant.
             */
            virtual void useAutomaticBinaryPredictor() {
                this->getBinaryPredictorConfig().set(std::make_unique<AutomaticBinaryPredictorConfig>(
                  this->getClassificationLossConfig(), this->getParallelPredictionConfig()));
            }
    };
}
