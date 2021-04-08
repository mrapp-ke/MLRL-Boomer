/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "seco/statistics/statistics_label_wise.hpp"
#include "common/input/label_matrix_c_contiguous.hpp"


namespace seco {

    /**
     * A factory that allows to create new instances of the class `DenseLabelWiseStatistics`.
     */
    class DenseLabelWiseStatisticsFactory final : public ILabelWiseStatisticsFactory {

        private:

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            const CContiguousLabelMatrix& labelMatrix_;

        public:

            /**
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             * @param labelMatrix               A reference to an object of type `CContiguousLabelMatrix` that provides
             *                                  random access to the labels of the training examples
             */
            DenseLabelWiseStatisticsFactory(std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                            const CContiguousLabelMatrix& labelMatrix);

            std::unique_ptr<ILabelWiseStatistics> create() const override;

    };

}
