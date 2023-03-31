/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/probability_function_marginal.hpp"
#include "common/data/view_vector.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to transform the regression scores that are predicted an example
     * into a joint probability that corresponds to the chance of a label vector being correct.
     */
    class IJointProbabilityFunction {
        public:

            virtual ~IJointProbabilityFunction() {};

            /**
             * Transforms the regression scores that are predicted an example into a joint probability that corresponds
             * to the chance of a given label vector being correct.
             *
             * @param scoresBegin   A `VectorConstView::const_iterator` to the beginning of the scores
             * @param scoresEnd     A `VectorConstView::const_iterator` to the end of the scores
             * @param labelVector   A reference to an object of type `LabelVector` the scores should be compared to
             * @return              The joint probability that corresponds to the chance of the given label vector being
             *                      correct
             */
            virtual float64 transformScoresIntoJointProbability(VectorConstView<float64>::const_iterator scoresBegin,
                                                                VectorConstView<float64>::const_iterator scoresEnd,
                                                                const LabelVector& labelVector) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `IJointProbabilityFunction`.
     */
    class IJointProbabilityFunctionFactory {
        public:

            virtual ~IJointProbabilityFunctionFactory() {};

            /**
             * Creates and returns a new object of the type `IJointProbabilityFunction`.
             *
             * @return An unique pointer to an object of type `IJointProbabilityFunction` that has been created
             */
            virtual std::unique_ptr<IJointProbabilityFunction> create() const = 0;
    };

}
