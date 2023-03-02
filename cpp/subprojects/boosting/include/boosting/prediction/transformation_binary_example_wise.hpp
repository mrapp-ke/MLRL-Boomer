/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/transformation_binary.hpp"
#include "common/measures/measure_distance.hpp"
#include "common/prediction/label_vector_set.hpp"

namespace boosting {

    /**
     * An implementation of the class `IBinaryTransformation` that transforms real-valued predictions into binary
     * predictions by comparing the real-valued predictions to the known label vectors according to a certain distance
     * measure and picking the closest one.
     */
    class ExampleWiseBinaryTransformation final : public IBinaryTransformation {
        private:

            const LabelVectorSet& labelVectorSet_;

            std::unique_ptr<IDistanceMeasure> distanceMeasurePtr_;

        public:

            /**
             * @param labelVectorSet        A reference to an object of type `LabelVectorSet` that stores all known
             *                              label vectors
             * @param distanceMeasurePtr    An unique pointer to an object of type `IDistanceMeasure` that implements
             *                              the distance measure for comparing real-valued predictions to known label
             *                              vectors
             */
            ExampleWiseBinaryTransformation(const LabelVectorSet& labelVectorSet,
                                            std::unique_ptr<IDistanceMeasure> distanceMeasurePtr);

            void apply(CContiguousConstView<float64>::value_const_iterator realBegin,
                       CContiguousConstView<float64>::value_const_iterator realEnd,
                       CContiguousView<uint8>::value_iterator predictionBegin,
                       CContiguousView<uint8>::value_iterator predictionEnd) const override;

            void apply(CContiguousConstView<float64>::value_const_iterator realBegin,
                       CContiguousConstView<float64>::value_const_iterator realEnd,
                       BinaryLilMatrix::row predictionRow) const override;
    };

}
