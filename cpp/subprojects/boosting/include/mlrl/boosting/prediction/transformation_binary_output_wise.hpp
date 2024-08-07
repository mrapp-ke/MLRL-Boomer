/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/discretization_function.hpp"
#include "mlrl/boosting/prediction/transformation_binary.hpp"

#include <memory>

namespace boosting {

    /**
     * An implementation of the class `IBinaryTransformation` that transforms the individual scores or probability
     * estimates that are predicted for each label into binary predictions via element-wise application of an
     * `IDiscretizationFunction`.
     */
    class OutputWiseBinaryTransformation final : public IBinaryTransformation {
        private:

            std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr_;

        public:

            /**
             * @param discretizationFunctionPtr An unique pointer to an object of type `IDiscretizationFunction` that
             *                                  should be used to discretize scores
             */
            OutputWiseBinaryTransformation(std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr);

            void apply(View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd,
                       View<uint8>::iterator predictionBegin, View<uint8>::iterator predictionEnd) const override;

            void apply(View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd,
                       BinaryLilMatrix::row predictionRow) const override;
    };

}
