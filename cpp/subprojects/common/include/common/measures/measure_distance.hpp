/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_vector.hpp"
#include "common/prediction/label_vector_set.hpp"
#include "common/prediction/probability_calibration_joint.hpp"

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

        /**
         * Searches among the label vectors contained in a `LabelVectorSet` and returns the one that is closest to the
         * scores that are predicted for an example.
         *
         * @param labelVectorSet        A reference to an object of type `LabelVectorSet` that contains the label
         *                              vectors
         * @param scoresBegin           A `VectorConstView::const_iterator` to the beginning of the predicted scores
         * @param scoresEnd             A `VectorConstView::const_iterator` to the end of the predicted scores
         * @return                      A reference to an object of type `LabelVector` that has been found
         */
        virtual const LabelVector& getClosestLabelVector(const LabelVectorSet& labelVectorSet,
                                                         VectorConstView<float64>::const_iterator scoresBegin,
                                                         VectorConstView<float64>::const_iterator scoresEnd) const {
            LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet.cbegin();
            LabelVectorSet::frequency_const_iterator frequencyIterator = labelVectorSet.frequencies_cbegin();
            uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();
            const LabelVector* closestLabelVector = labelVectorIterator[0].get();
            uint32 maxFrequency = frequencyIterator[0];
            float64 minDistance = this->measureDistance(*closestLabelVector, scoresBegin, scoresEnd);

            for (uint32 i = 1; i < numLabelVectors; i++) {
                const LabelVector& labelVector = *labelVectorIterator[i];
                uint32 frequency = frequencyIterator[i];
                float64 distance = this->measureDistance(labelVector, scoresBegin, scoresEnd);

                if (distance < minDistance || (distance == minDistance && frequency > maxFrequency)) {
                    closestLabelVector = &labelVector;
                    maxFrequency = frequency;
                    minDistance = distance;
                }
            }

            return *closestLabelVector;
        }
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
         * @param marginalProbabilityCalibrationModel   A reference to an object of type
         *                                              `IMarginalProbabilityCalibrationModel` that should be used for
         *                                              the calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel      A reference to an object of type
         *                                              `IJointProbabilityCalibrationModel` that should be used for the
         *                                              calibration of joint probabilities
         * @return                                      An unique pointer to an object of type `IDistanceMeasure` that
         *                                              has been created
         */
        virtual std::unique_ptr<IDistanceMeasure> createDistanceMeasure(
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const = 0;
};
