/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/data/view_vector.hpp"

#include <memory>

/**
 * Defines an interface for all measures that may be used to compare predictions for individual examples to the
 * corresponding ground truth labels in order to obtain a distance.
 */
class IDistanceMeasure {
    public:

        virtual ~IDistanceMeasure() {};

        /**
         * Calculates and returns the distance between the predicted scores for a single example and the corresponding
         * ground truth labels.
         *
         * @param relevantLabelIndices  A reference to an object of type `VectorConstView` that provides access to the
         *                              indices of the relevant labels according to the ground truth
         * @param scoresBegin           A `VectorConstView::const_iterator` to the beginning of the predicted scores
         * @param scoresEnd             A `VectorConstView::const_iterator` to the end of the predicted scores
         * @return                      The distance that has been calculated
         */
        virtual float64 measureDistance(const VectorConstView<uint32>& relevantLabelIndices,
                                        VectorConstView<float64>::const_iterator scoresBegin,
                                        VectorConstView<float64>::const_iterator scoresEnd) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IDistanceMeasure`.
 */
class IDistanceMeasureFactory {
    public:

        virtual ~IDistanceMeasureFactory() {};

        /**
         * Creates and returns a new object of type `IDistanceMeasure`.
         *
         * @return An unique pointer to an object of type `IDistanceMeasure` that has been created
         */
        virtual std::unique_ptr<IDistanceMeasure> createDistanceMeasure() const = 0;
};
