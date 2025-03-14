/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/transformation_binary.hpp"
#include "mlrl/common/measures/measure_distance.hpp"
#include "mlrl/common/prediction/label_vector_set.hpp"

#include <memory>

namespace boosting {

    /**
     * An implementation of the class `IBinaryTransformation` that transforms scores into binary predictions by
     * comparing the scores to the known label vectors according to a certain distance measure and picking the closest
     * one.
     */
    class ExampleWiseBinaryTransformation final : public IBinaryTransformation {
        private:

            const LabelVectorSet& labelVectorSet_;

            const std::unique_ptr<IDistanceMeasure<float64>> distanceMeasurePtr_;

        public:

            /**
             * @param labelVectorSet        A reference to an object of type `LabelVectorSet` that stores all known
             *                              label vectors
             * @param distanceMeasurePtr    An unique pointer to an object of type `IDistanceMeasure` that implements
             *                              the distance measure for comparing scores to known label vectors
             */
            ExampleWiseBinaryTransformation(const LabelVectorSet& labelVectorSet,
                                            std::unique_ptr<IDistanceMeasure<float64>> distanceMeasurePtr);

            void apply(View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd,
                       View<uint8>::iterator predictionBegin, View<uint8>::iterator predictionEnd) const override;

            void apply(View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd,
                       BinaryLilMatrix::row predictionRow) const override;
    };

}
