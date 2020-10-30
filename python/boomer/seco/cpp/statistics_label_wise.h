/**
 * Provides classes that allow to store the elements of confusion matrices that are computed independently for each
 * label.
 *
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/input_data.h"
#include "../../common/cpp/statistics.h"
#include "rule_evaluation_label_wise.h"
#include "statistics.h"
#include <memory>


namespace seco {

    /**
     * An abstract base class for all classes that allow to store the elements of confusion matrices that are computed
     * independently for each label.
     */
    class AbstractLabelWiseStatistics : public AbstractCoverageStatistics {

        protected:

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

        public:

            /**
             * @param numStatistics             The number of statistics
             * @param numLabels                 The number of labels
             * @param sumUncoveredLabels        The sum of weights of all labels that remain to be covered, initially
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             */
            AbstractLabelWiseStatistics(uint32 numStatistics, uint32 numLabels, float64 sumUncoveredLabels,
                                        std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr);

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             *
             * @param ruleEvaluationFactoryPtr A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                 to be set
             */
            void setRuleEvaluationFactory(std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr);

    };

    /**
     * Defines an interface for all classes that allow to create new instances of the class
     * `AbstractLabelWiseStatistics`.
     */
    class ILabelWiseStatisticsFactory {

        public:

            virtual ~ILabelWiseStatisticsFactory() { };

            /**
             * Creates a new instance of the class `AbstractLabelWiseStatistics`.
             *
             * @return An unique pointer to an object of type `AbstractLabelWiseStatistics` that has been created
             */
            virtual std::unique_ptr<AbstractLabelWiseStatistics> create() const = 0;

    };

    /**
     * A factory that allows to create new instances of the class `DenseLabelWiseStatisticsImpl`.
     */
    class DenseLabelWiseStatisticsFactoryImpl : virtual public ILabelWiseStatisticsFactory {

        private:

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             */
            DenseLabelWiseStatisticsFactoryImpl(
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr);

            std::unique_ptr<AbstractLabelWiseStatistics> create() const override;

    };

}
