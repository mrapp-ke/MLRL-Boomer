/**
 * Provides classes that allow to store gradients and Hessians that are calculated according to a (decomposable) loss
 * function that is applied label-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/statistics/statistics.h"
#include "../losses/loss_label_wise.h"
#include "../rule_evaluation_label_wise.h"


namespace boosting {

    /**
     * Defines an interface for all classes that store gradients and Hessians that have been calculated according to a
     * differentiable loss function that is applied label-wise.
     */
    class ILabelWiseStatistics : virtual public IStatistics {

        public:

            virtual ~ILabelWiseStatistics() { };

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             *
             * @param ruleEvaluationFactoryPtr A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                 to be set
             */
            virtual void setRuleEvaluationFactory(
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) = 0;

    };

    /**
     * Defines an interface for all classes that allow to create new instances of the type `ILabelWiseStatistics`.
     */
    class ILabelWiseStatisticsFactory {

        public:

            virtual ~ILabelWiseStatisticsFactory() { };

            /**
             * Creates a new instance of the type `ILabelWiseStatistics`.
             *
             * @return An unique pointer to an object of type `ILabelWiseStatistics` that has been created
             */
            virtual std::unique_ptr<ILabelWiseStatistics> create() const = 0;

    };

    /**
     * A factory that allows to create new instances of the class `LabelWiseStatistics`.
     */
    class DenseLabelWiseStatisticsFactory : public ILabelWiseStatisticsFactory {

        private:

            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `ILabelWiseLoss`, representing the
             *                                  loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used to calculate
             *                                  the predictions, as well as corresponding quality scores, of rules
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             */
            DenseLabelWiseStatisticsFactory(std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
                                            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr);

            std::unique_ptr<ILabelWiseStatistics> create() const override;

    };

}
