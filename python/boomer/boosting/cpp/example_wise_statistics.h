/**
 * Provides classes that allow to store gradients and Hessians that are calculated according to a (non-decomposable)
 * loss function that is applied example-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/statistics.h"
#include "example_wise_rule_evaluation.h"
#include "example_wise_losses.h"
#include "statistics.h"
#include "lapack.h"


namespace boosting {

    /**
     * An abstract base class for all classes that allow to store gradients and Hessians that are calculated according
     * to a differentiable loss function that is applied example-wise.
     */
    class AbstractExampleWiseStatistics : public AbstractGradientStatistics {

        protected:

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

        public:

            /**
             * @param numStatistics             The number of statistics
             * @param numLabels                 The number of labels
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory` that allows to create instances of
             *                                  the class that is used for calculating the predictions, as well as
             *                                  corresponding quality scores, of rules
             */
            AbstractExampleWiseStatistics(uint32 numStatistics, uint32 numLabels,
                                          std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr);

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             *
             * @param ruleEvaluationFactoryPtr A shared pointer to an object of type `IExampleWiseRuleFactoryEvaluation`
             *                                 to be set
             */
            void setRuleEvaluationFactory(std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr);

    };

    /**
     * Defines an interface for all classes that allow to create new instances of the class
     * `AbstractExampleWiseStatistics`.
     */
    class IExampleWiseStatisticsFactory {

        public:

            virtual ~IExampleWiseStatisticsFactory() { };

            /**
             * Creates a new instance of the class `AbstractExampleWiseStatistics`.
             *
             * @return An unique pointer to an object of type `AbstractExampleWiseStatistics` that has been created
             */
            virtual std::unique_ptr<AbstractExampleWiseStatistics> create() const = 0;

    };

    /**
     * A factory that allows to create new instances of the class `DenseExampleWiseStatisticsImpl`.
     */
    class DenseExampleWiseStatisticsFactoryImpl : virtual public IExampleWiseStatisticsFactory {

        private:

            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `IExampleWiseLoss`, representing
             *                                  the loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory`, to be used for calculating the
             *                                  predictions, as well as corresponding quality scores, of rules
             * @param lapackPtr                 An unique pointer to an object of type `Lapack` that allows to execute
             *                                  different Lapack routines
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             */
            DenseExampleWiseStatisticsFactoryImpl(
                    std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                    std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                    std::unique_ptr<Lapack> lapackPtr, std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr);

            std::unique_ptr<AbstractExampleWiseStatistics> create() const override;

    };

}
