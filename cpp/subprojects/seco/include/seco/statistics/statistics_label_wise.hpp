/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/input/label_matrix_csr.hpp"
#include "seco/statistics/statistics.hpp"
#include "seco/rule_evaluation/rule_evaluation_label_wise.hpp"
#include <memory>


namespace seco {

    /**
     * Defines an interface for all classes that allow to store the elements of confusion matrices that are computed
     * independently for each label.
     */
    class ILabelWiseStatistics : public ICoverageStatistics {

        public:

            virtual ~ILabelWiseStatistics() { };

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             *
             * @param ruleEvaluationFactory A reference to an object of type `ILabelWiseRuleEvaluationFactory` to be set
             */
            virtual void setRuleEvaluationFactory(const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory) = 0;

    };

    /**
     * Defines an interface for all classes that allow to create new instances of the class ILabelWiseStatistics`.
     */
    class ILabelWiseStatisticsFactory {

        public:

            virtual ~ILabelWiseStatisticsFactory() { };

            /**
             * Creates a new instance of the class `ILabelWiseStatistics`, based on a matrix that provides random access
             * to the labels of the training examples.
             *
             * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides random
             *                      access to the labels of the training examples
             * @return              An unique pointer to an object of type `ILabelWiseStatistics` that has been created
             */
            virtual std::unique_ptr<ILabelWiseStatistics> create(const CContiguousLabelMatrix& labelMatrix) const = 0;

            /**
             * Creates a new instance of the class `ILabelWiseStatistics`, based on a matrix that provides row-wise
             * access to the labels of the training examples.
             *
             * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides row-wise
             *                      access to the labels of the training examples
             * @return              An unique pointer to an object of type `ILabelWiseStatistics` that has been created
             */
            virtual std::unique_ptr<ILabelWiseStatistics> create(const CsrLabelMatrix& labelMatrix) const = 0;

    };

}
