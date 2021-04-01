/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/statistics/statistics_label_wise.hpp"
#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * A factory that allows to create new instances of the type `ILabelWiseStatistics` that use sparse data structures.
     */
    class SparseLabelWiseStatisticsFactory final : public ILabelWiseStatisticsFactory {

        private:

            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::shared_ptr<CsrLabelMatrix> labelMatrixPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `ILabelWiseLoss`, representing the
             *                                  loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used to calculate
             *                                  the predictions, as well as corresponding quality scores, of rules
             * @param labelMatrixPtr            A shared pointer to an object of type `CsrLabelMatrix` that provides
             *                                  row-wise to the labels of the training examples
             * @param numThreads                The number of CPU threads to be used to calculate the initial statistics
             *                                  in parallel. Must be at least 1
             */
            SparseLabelWiseStatisticsFactory(std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
                                             std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                             std::shared_ptr<CsrLabelMatrix> labelMatrixPtr, uint32 numThreads);

            std::unique_ptr<ILabelWiseStatistics> create() const override;

    };

}
