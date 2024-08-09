/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/probability_function_joint.hpp"
#include "mlrl/boosting/prediction/transformation_probability.hpp"

#include <memory>

namespace boosting {

    /**
     * An implementation of the class `IProbabilityTransformation` that transforms aggregated scores into marginalized
     * probability estimates.
     */
    class MarginalizedProbabilityTransformation final : public IProbabilityTransformation {
        private:

            const LabelVectorSet& labelVectorSet_;

            const std::unique_ptr<IJointProbabilityFunction> jointProbabilityFunctionPtr_;

        public:

            /**
             * @param labelVectorSet                A reference to an object of type `LabelVectorSet` that stores all
             *                                      known label vectors
             * @param jointProbabilityFunctionPtr   An unique pointer to an object of type `JointProbabilityFunction`
             *                                      that should be used to transform scores that are predicted for
             *                                      individual labels into probabilities
             */
            MarginalizedProbabilityTransformation(
              const LabelVectorSet& labelVectorSet,
              std::unique_ptr<IJointProbabilityFunction> jointProbabilityFunctionPtr);

            void apply(View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd,
                       View<float64>::iterator probabilitiesBegin,
                       View<float64>::iterator probabilitiesEnd) const override;
    };

}
