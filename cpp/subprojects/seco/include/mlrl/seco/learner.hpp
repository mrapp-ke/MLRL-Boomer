/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/learner_classification.hpp"
#include "mlrl/seco/heuristics/heuristic_accuracy.hpp"
#include "mlrl/seco/heuristics/heuristic_f_measure.hpp"
#include "mlrl/seco/heuristics/heuristic_laplace.hpp"
#include "mlrl/seco/heuristics/heuristic_m_estimate.hpp"
#include "mlrl/seco/heuristics/heuristic_precision.hpp"
#include "mlrl/seco/heuristics/heuristic_recall.hpp"
#include "mlrl/seco/heuristics/heuristic_wra.hpp"
#include "mlrl/seco/lift_functions/lift_function_kln.hpp"
#include "mlrl/seco/lift_functions/lift_function_no.hpp"
#include "mlrl/seco/lift_functions/lift_function_peak.hpp"
#include "mlrl/seco/prediction/predictor_binary_output_wise.hpp"
#include "mlrl/seco/rule_evaluation/head_type_partial.hpp"
#include "mlrl/seco/rule_evaluation/head_type_single.hpp"
#include "mlrl/seco/stopping/stopping_criterion_coverage.hpp"

#include <memory>
#include <utility>

namespace seco {

    /**
     * Defines an interface for all rule learners that make use of the separate-and-conquer (SeCo) paradigm.
     */
    class MLRLSECO_API ISeCoRuleLearner : virtual public IClassificationRuleLearner {
        public:

            /**
             * Defines an interface for all classes that allow to configure a rule learner that makes use of the
             * separate-and-conquer (SeCo) paradigm.
             */
            class IConfig : virtual public IClassificationRuleLearner::IConfig {
                public:

                    virtual ~IConfig() override {}

                    /**
                     * Returns an unique pointer to the configuration of the stopping criterion that stops the induction
                     * of rules as soon as the sum of the weights of the uncovered labels is smaller or equal to a
                     * certain threshold.
                     *
                     * @return A reference to an unique pointer of type `CoverageStoppingCriterionConfig` that stores
                     *         the configuration of the stopping criterion that stops the induction of rules as soon as
                     *         the sum of the weights of the uncovered labels is smaller or equal to a certain threshold
                     *         or a null pointer, if no such stopping criterion should be used
                     */
                    virtual std::unique_ptr<CoverageStoppingCriterionConfig>&
                      getCoverageStoppingCriterionConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the rule heads that should be induced by the
                     * rule learner.
                     *
                     * @return A reference to an unique pointer of type `IHeadConfig` that stores the configuration of
                     *         the rule heads
                     */
                    virtual std::unique_ptr<IHeadConfig>& getHeadConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the heuristic for learning rules.
                     *
                     * @return A reference to an unique pointer of type `IHeuristicConfig` that stores the configuration
                     *         of the heuristic for learning rules
                     */
                    virtual std::unique_ptr<IHeuristicConfig>& getHeuristicConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the heuristic for pruning rules.
                     *
                     * @return A reference to an unique pointer of type `IHeuristicConfig` that stores the configuration
                     *         of the heuristic for pruning rules
                     */
                    virtual std::unique_ptr<IHeuristicConfig>& getPruningHeuristicConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the lift function that affects the quality of
                     * rules, depending on the number of labels for which they predict.
                     *
                     * @return A reference to an unique pointer of type `ILiftFunctionConfig` that stores the
                     *         configuration of the lift function that affects the quality of rules, depending on the
                     *         number of labels for which they predict
                     */
                    virtual std::unique_ptr<ILiftFunctionConfig>& getLiftFunctionConfigPtr() = 0;
            };

            virtual ~ISeCoRuleLearner() override {}

            /**
             * Defines an interface for all classes that allow to configure a rule learner to not use any stopping
             * criterion that stops the induction of rules as soon as the sum of the weights of the uncovered labels is
             * smaller or equal to a certain threshold.
             */
            class INoCoverageStoppingCriterionMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~INoCoverageStoppingCriterionMixin() override {}

                    /**
                     * Configures the rule learner to not use any stopping criterion that stops the induction of rules
                     * as soon as the sum of the weights of the uncovered labels is smaller or equal to a certain
                     * threshold.
                     */
                    virtual void useNoCoverageStoppingCriterion() {
                        std::unique_ptr<CoverageStoppingCriterionConfig>& coverageStoppingCriterionConfigPtr =
                          this->getCoverageStoppingCriterionConfigPtr();
                        coverageStoppingCriterionConfigPtr = nullptr;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a stopping criterion
             * that stops the induction of rules as soon as the sum of the weights of the uncovered labels is smaller or
             * equal to a certain threshold.
             */
            class ICoverageStoppingCriterionMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~ICoverageStoppingCriterionMixin() override {}

                    /**
                     * Configures the rule learner to use a stopping criterion that stops the induction of rules as soon
                     * as the sum of the weights of the uncovered labels is smaller or equal to a certain threshold.
                     *
                     * @return A reference to an object of type `ICoverageStoppingCriterionConfig` that allows further
                     *         configuration of the stopping criterion
                     */
                    virtual ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion() {
                        std::unique_ptr<CoverageStoppingCriterionConfig>& coverageStoppingCriterionConfigPtr =
                          this->getCoverageStoppingCriterionConfigPtr();
                        std::unique_ptr<CoverageStoppingCriterionConfig> ptr =
                          std::make_unique<CoverageStoppingCriterionConfig>();
                        ICoverageStoppingCriterionConfig& ref = *ptr;
                        coverageStoppingCriterionConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to induce rules with
             * single-output heads that predict for a single output.
             */
            class ISingleOutputHeadMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~ISingleOutputHeadMixin() override {}

                    /**
                     * Configures the rule learner to induce rules with single-output heads that predict for a single
                     * output.
                     */
                    virtual void useSingleOutputHeads() {
                        std::unique_ptr<IHeadConfig>& headConfigPtr = this->getHeadConfigPtr();
                        headConfigPtr = std::make_unique<SingleOutputHeadConfig>(this->getHeuristicConfigPtr(),
                                                                                 this->getPruningHeuristicConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to induce rules with partial
             * heads.
             */
            class IPartialHeadMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IPartialHeadMixin() override {}

                    /**
                     * Configures the rule learner to induce rules with partial heads that predict for a subset of the
                     * available labels.
                     */
                    virtual void usePartialHeads() {
                        std::unique_ptr<IHeadConfig>& headConfigPtr = this->getHeadConfigPtr();
                        headConfigPtr = std::make_unique<PartialHeadConfig>(this->getHeuristicConfigPtr(),
                                                                            this->getPruningHeuristicConfigPtr(),
                                                                            this->getLiftFunctionConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to not use a lift function.
             */
            class INoLiftFunctionMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~INoLiftFunctionMixin() override {}

                    /**
                     * Configures the rule learner to not use a lift function.
                     */
                    virtual void useNoLiftFunction() {
                        std::unique_ptr<ILiftFunctionConfig>& liftFunctionConfigPtr = this->getLiftFunctionConfigPtr();
                        liftFunctionConfigPtr = std::make_unique<NoLiftFunctionConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a lift function that
             * monotonously increases until a certain number of labels, where the maximum lift is reached, and
             * monotonously decreases afterwards.
             */
            class IPeakLiftFunctionMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IPeakLiftFunctionMixin() override {}

                    /**
                     * Configures the rule learner to use a lift function that monotonously increases until a certain
                     * number of labels, where the maximum lift is reached, and monotonously decreases afterwards.
                     *
                     * @return A reference to an object of type `IPeakLiftFunctionConfig` that allows further
                     *         configuration of the lift function
                     */
                    virtual IPeakLiftFunctionConfig& usePeakLiftFunction() {
                        std::unique_ptr<ILiftFunctionConfig>& liftFunctionConfigPtr = this->getLiftFunctionConfigPtr();
                        std::unique_ptr<PeakLiftFunctionConfig> ptr = std::make_unique<PeakLiftFunctionConfig>();
                        IPeakLiftFunctionConfig& ref = *ptr;
                        liftFunctionConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a lift function that
             * monotonously increases according to the natural logarithm of the number of labels for which a rule
             * predicts.
             */
            class IKlnLiftFunctionMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IKlnLiftFunctionMixin() override {}

                    /**
                     * Configures the rule learner to use a lift function that monotonously increases according to the
                     * natural logarithm of the number of labels for which a rule predicts.
                     *
                     * @return A reference to an object of type `IKlnLiftFunctionConfig` that allows further
                     *         configuration of the lift function
                     */
                    virtual IKlnLiftFunctionConfig& useKlnLiftFunction() {
                        std::unique_ptr<ILiftFunctionConfig>& liftFunctionConfigPtr = this->getLiftFunctionConfigPtr();
                        std::unique_ptr<KlnLiftFunctionConfig> ptr = std::make_unique<KlnLiftFunctionConfig>();
                        IKlnLiftFunctionConfig& ref = *ptr;
                        liftFunctionConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Accuracy"
             * heuristic for learning rules.
             */
            class IAccuracyHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IAccuracyHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "Accuracy" heuristic for learning rules.
                     */
                    virtual void useAccuracyHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        heuristicConfigPtr = std::make_unique<AccuracyConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Accuracy"
             * heuristic for pruning rules.
             */
            class IAccuracyPruningHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IAccuracyPruningHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "Accuracy" heuristic for pruning rules.
                     */
                    virtual void useAccuracyPruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        pruningHeuristicConfigPtr = std::make_unique<AccuracyConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "F-Measure"
             * heuristic for learning rules.
             */
            class IFMeasureHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IFMeasureHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IFMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IFMeasureConfig& useFMeasureHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
                        IFMeasureConfig& ref = *ptr;
                        heuristicConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "F-Measure"
             * heuristic for pruning rules.
             */
            class IFMeasurePruningHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IFMeasurePruningHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IFMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IFMeasureConfig& useFMeasurePruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
                        IFMeasureConfig& ref = *ptr;
                        pruningHeuristicConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "M-Estimate"
             * heuristic for learning rules.
             */
            class IMEstimateHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IMEstimateHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IMEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IMEstimateConfig& useMEstimateHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
                        IMEstimateConfig& ref = *ptr;
                        heuristicConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "M-Estimate"
             * heuristic for pruning rules.
             */
            class IMEstimatePruningHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IMEstimatePruningHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IMEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IMEstimateConfig& useMEstimatePruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
                        IMEstimateConfig& ref = *ptr;
                        pruningHeuristicConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Laplace"
             * heuristic for learning rules.
             */
            class ILaplaceHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~ILaplaceHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for learning rules.
                     */
                    virtual void useLaplaceHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        heuristicConfigPtr = std::make_unique<LaplaceConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Laplace"
             * heuristic for pruning rules.
             */
            class ILaplacePruningHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~ILaplacePruningHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for pruning rules.
                     */
                    virtual void useLaplacePruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        pruningHeuristicConfigPtr = std::make_unique<LaplaceConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Precision"
             * heuristic for learning rules.
             */
            class IPrecisionHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IPrecisionHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "Precision" heuristic for learning rules.
                     */
                    virtual void usePrecisionHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        heuristicConfigPtr = std::make_unique<PrecisionConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Precision"
             * heuristic for pruning rules.
             */
            class IPrecisionPruningHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IPrecisionPruningHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "Precision" heuristic for pruning rules.
                     */
                    virtual void usePrecisionPruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        pruningHeuristicConfigPtr = std::make_unique<PrecisionConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Recall" heuristic
             * for pruning rules.
             */
            class IRecallHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IRecallHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for learning rules.
                     */
                    virtual void useRecallHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        heuristicConfigPtr = std::make_unique<RecallConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Recall" heuristic
             * for pruning rules.
             */
            class IRecallPruningHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IRecallPruningHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for pruning rules.
                     */
                    virtual void useRecallPruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        pruningHeuristicConfigPtr = std::make_unique<RecallConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Weighted Relative
             * Accuracy" (WRA) heuristic for learning rules.
             */
            class IWraHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IWraHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" (WRA) heuristic for learning
                     * rules.
                     */
                    virtual void useWraHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        heuristicConfigPtr = std::make_unique<WraConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Weighted Relative
             * Accuracy" (WRA) heuristic for pruning rules.
             */
            class IWraPruningHeuristicMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IWraPruningHeuristicMixin() override {}

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" (WRA) heuristic for pruning
                     * rules.
                     */
                    virtual void useWraPruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        pruningHeuristicConfigPtr = std::make_unique<WraConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor for
             * predicting whether individual labels of given query examples are relevant or irrelevant by processing
             * rules of an existing rule-based model in the order they have been learned. If a rule covers an example,
             * its prediction is applied to each label individually.
             */
            class IOutputWiseBinaryPredictionMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IOutputWiseBinaryPredictionMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor for predicting whether individual labels of given
                     * query examples are relevant or irrelevant by processing rules of an existing rule-based model in
                     * the order they have been learned. If a rule covers an example, its prediction is applied to each
                     * label individually.
                     */
                    virtual void useOutputWiseBinaryPredictor() {
                        std::unique_ptr<IBinaryPredictorConfig>& binaryPredictorConfigPtr =
                          this->getBinaryPredictorConfigPtr();
                        binaryPredictorConfigPtr =
                          std::make_unique<OutputWiseBinaryPredictorConfig>(this->getParallelPredictionConfigPtr());
                    }
            };
    };
}
