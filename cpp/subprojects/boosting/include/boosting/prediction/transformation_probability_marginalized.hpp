/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/probability_function.hpp"
#include "boosting/prediction/transformation_probability.hpp"
#include "common/prediction/label_vector_set.hpp"

namespace boosting {

    /**
     * An implementation of the class `IProbabilityTransformation` that transforms aggregated scores into marginalized
     * probability estimates.
     */
    class MarginalizedProbabilityTransformation final : public IProbabilityTransformation {
        private:

            const LabelVectorSet& labelVectorSet_;

            std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr_;

        public:

            MarginalizedProbabilityTransformation(const LabelVectorSet& labelVectorSet,
                                                  std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr);

            void apply(CContiguousView<float64>::value_iterator scoresBegin,
                       CContiguousView<float64>::value_iterator scoresEnd) const override;
    };

}
