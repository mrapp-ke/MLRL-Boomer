#include "boosting/rule_evaluation/head_type_complete.hpp"


namespace boosting {

    std::unique_ptr<IStatisticsProviderFactory> CompleteHeadConfig::configure() const {
        /*
        std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr,
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads

        return DenseExampleWiseStatisticsProviderFactory(loss_factory, evaluation_measure_factory,
                                                                 default_rule_evaluation_factory,
                                                                 regular_rule_evaluation_factory,
                                                                 pruning_rule_evaluation_factory, num_threads)
        */
        // TODO
        return nullptr;
    }

}
