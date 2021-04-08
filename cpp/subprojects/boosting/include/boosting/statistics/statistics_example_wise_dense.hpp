/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "boosting/statistics/statistics_example_wise.hpp"
#include "boosting/losses/loss_example_wise.hpp"


namespace boosting {

    /**
     * A factory that allows to create new instances of the class `ExampleWiseStatistics`.
     */
    class DenseExampleWiseStatisticsFactory final : public IExampleWiseStatisticsFactory {

        private:

            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `IExampleWiseLoss`, representing
             *                                  the loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory`, to be used for calculating the
             *                                  predictions, as well as corresponding quality scores, of rules
             * @param numThreads                The number of CPU threads to be used to calculate the initial statistics
             *                                  in parallel. Must be at least 1
             */
            DenseExampleWiseStatisticsFactory(
                    std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                    std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, uint32 numThreads);

            std::unique_ptr<IExampleWiseStatistics> create(const CContiguousLabelMatrix& labelMatrix) const override;

            std::unique_ptr<IExampleWiseStatistics> create(const CsrLabelMatrix& labelMatrix) const override;

    };

}
