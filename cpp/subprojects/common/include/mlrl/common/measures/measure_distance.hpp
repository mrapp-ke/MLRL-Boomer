/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"
#include "mlrl/common/prediction/label_vector_set.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"

#include <memory>

/**
 * Defines an interface for all measures that may be used to compare scores that are predicted for individual examples
 * to the corresponding ground truth labels in order to obtain a distance.
 *
 * @tparam ScoreType The type of the predicted scores
 */
template<typename ScoreType>
class IDistanceMeasure {
    public:

        virtual ~IDistanceMeasure() {}

        /**
         * Calculates and returns the distance between the predicted scores for a single example and a label vector.
         *
         * @param labelVectorIndex  The index of the label vector, the scores should be compared to
         * @param labelVector       A reference to an object of type `LabelVector`, the scores should be compared to
         * @param scoresBegin       An iterator to the beginning of the predicted scores
         * @param scoresEnd         An iterator to the end of the predicted scores
         * @return                  The distance that has been calculated
         */
        virtual ScoreType measureDistance(uint32 labelVectorIndex, const LabelVector& labelVector,
                                          typename View<ScoreType>::const_iterator scoresBegin,
                                          typename View<ScoreType>::const_iterator scoresEnd) const = 0;

        /**
         * Searches among the label vectors contained in a `LabelVectorSet` and returns the one that is closest to the
         * scores that are predicted for an example.
         *
         * @param labelVectorSet        A reference to an object of type `LabelVectorSet` that contains the label
         *                              vectors
         * @param scoresBegin           An iterator to the beginning of the predicted scores
         * @param scoresEnd             An iterator to the end of the predicted scores
         * @return                      A reference to an object of type `LabelVector` that has been found
         */
        virtual const LabelVector& getClosestLabelVector(const LabelVectorSet& labelVectorSet,
                                                         typename View<ScoreType>::const_iterator scoresBegin,
                                                         typename View<ScoreType>::const_iterator scoresEnd) const {
            LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet.cbegin();
            LabelVectorSet::frequency_const_iterator frequencyIterator = labelVectorSet.frequencies_cbegin();
            uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();
            const LabelVector* closestLabelVector = labelVectorIterator[0].get();
            uint32 maxFrequency = frequencyIterator[0];
            ScoreType minDistance = this->measureDistance(0, *closestLabelVector, scoresBegin, scoresEnd);

            for (uint32 i = 1; i < numLabelVectors; i++) {
                const LabelVector& labelVector = *labelVectorIterator[i];
                uint32 frequency = frequencyIterator[i];
                ScoreType distance = this->measureDistance(i, labelVector, scoresBegin, scoresEnd);

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
 *
 * @tparam ScoreType The type of the predicted scores
 */
template<typename ScoreType>
class IDistanceMeasureFactory {
    public:

        virtual ~IDistanceMeasureFactory() {}

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
        virtual std::unique_ptr<IDistanceMeasure<ScoreType>> createDistanceMeasure(
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const = 0;
};
