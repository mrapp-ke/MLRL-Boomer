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

            const std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr_;

        public:

            /**
             * @param labelVectorSet            A reference to an object of type `LabelVectorSet` that stores all known
             *                                  label vectors
             * @param probabilityFunctionPtr    An unique pointer to an object of type `IProbabilityFunction` that
             *                                  should be used to transform aggregated scores into probability estimates
             */
            MarginalizedProbabilityTransformation(const LabelVectorSet& labelVectorSet,
                                                  std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr);

            void apply(CContiguousConstView<float64>::value_const_iterator scoresBegin,
                       CContiguousConstView<float64>::value_const_iterator scoresEnd,
                       CContiguousView<float64>::value_iterator probabilitiesBegin,
                       CContiguousView<float64>::value_iterator probabilitiesEnd) const override;
    };

}
