/**
 * Provides classes that allow to store the elements of confusion matrices that are computed independently for each
 * label.
 *
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "statistics_label_wise.h"
#include "../../../common/cpp/input/label_matrix.h"


namespace seco {

    /**
     * A factory that allows to create new instances of the class `DenseLabelWiseStatistics`.
     */
    class DenseLabelWiseStatisticsFactory final : public ILabelWiseStatisticsFactory {

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
            DenseLabelWiseStatisticsFactory(std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr);

            std::unique_ptr<ILabelWiseStatistics> create() const override;

    };

}
