/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/probability_function_marginal.hpp"
#include "common/data/vector_dense.hpp"
#include "common/math/math.hpp"
#include "common/measures/measure_distance.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to transform the regression scores that are predicted an example
     * into a joint probability that corresponds to the chance of a label vector being correct.
     */
    class IJointProbabilityFunction : public IDistanceMeasure {
        public:

            virtual ~IJointProbabilityFunction() {};

            /**
             * Transforms the regression scores that are predicted for an example into a joint probability that
             * corresponds to the chance of given ground truth labels being correct.
             *
             * @param relevantLabelIndices  A reference to an object of type `VectorConstView` that provides access to
             *                              the indices of the relevant labels according to the ground truth
             * @param scoresBegin           A `VectorConstView::const_iterator` to the beginning of the scores
             * @param scoresEnd             A `VectorConstView::const_iterator` to the end of the scores
             * @return                      The joint probability that corresponds to the chance of the given ground
             *                              truth labels being correct
             */
            virtual float64 transformScoresIntoJointProbability(
              const VectorConstView<uint32>& relevantLabelIndices, VectorConstView<float64>::const_iterator scoresBegin,
              VectorConstView<float64>::const_iterator scoresEnd) const = 0;

            /**
             * Transforms the regression scores that are predicted for an example into joint probabilities that
             * correspond to the chance of individual label vectors contained by a `LabelVectorSet` being correct.
             *
             * @param labelVectorSet    A reference to an object of type `LabelVectorSet` that contains the label
             *                          vectors the scores should be compared to
             * @param scoresBegin       A `VectorConstView::const_iterator` to the beginning of the scores
             * @param scoresEnd         A `VectorConstView::const_iterator` to the end of the scores
             * @return                  An unique pointer to an object of type `DenseVector` that stores the joint
             *                          probabilities that correspond to the chance of the given label vectors being
             *                          correct
             */
            virtual std::unique_ptr<DenseVector<float64>> transformScoresIntoJointProbabilities(
              const LabelVectorSet& labelVectorSet, VectorConstView<float64>::const_iterator scoresBegin,
              VectorConstView<float64>::const_iterator scoresEnd) const {
                uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();
                std::unique_ptr<DenseVector<float64>> jointProbabilityVectorPtr =
                  std::make_unique<DenseVector<float64>>(numLabelVectors);
                DenseVector<float64>::iterator jointProbabilityIterator = jointProbabilityVectorPtr->begin();
                float64 sumOfJointProbabilities = 0;
                uint32 i = 0;

                // Calculate joint probabilities...
                for (auto it = labelVectorSet.cbegin(); it != labelVectorSet.cend(); it++) {
                    const LabelVector& labelVector = *((*it).first);
                    float64 jointProbability =
                      this->transformScoresIntoJointProbability(labelVector, scoresBegin, scoresEnd);
                    sumOfJointProbabilities += jointProbability;
                    jointProbabilityIterator[i] = jointProbability;
                    i++;
                }

                // Normalize joint probabilities...
                for (i = 0; i < numLabelVectors; i++) {
                    float64 jointProbability = jointProbabilityIterator[i];
                    jointProbabilityIterator[i] = divideOrZero(jointProbability, sumOfJointProbabilities);
                }

                return jointProbabilityVectorPtr;
            }

            /**
             * @see `IDistanceMeasure::measureDistance`
             */
            float64 measureDistance(const VectorConstView<uint32>& relevantLabelIndices,
                                    VectorConstView<float64>::const_iterator scoresBegin,
                                    VectorConstView<float64>::const_iterator scoresEnd) const override final {
                return 1.0 - this->transformScoresIntoJointProbability(relevantLabelIndices, scoresBegin, scoresEnd);
            }
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `IJointProbabilityFunction`.
     */
    class IJointProbabilityFunctionFactory : public IDistanceMeasureFactory {
        public:

            virtual ~IJointProbabilityFunctionFactory() {};

            /**
             * Creates and returns a new object of the type `IJointProbabilityFunction`.
             *
             * @param marginalProbabilityCalibrationModel A reference to an object of type
             *                                            `IMarginalProbabilityCalibrationModel` that should be used for
             *                                            the calibration of marginal probabilities
             * @param jointProbabilityCalibrationModel    A reference to an object of type
             *                                            `IJointProbabilityCalibrationModel` that should be used for
             *                                            the calibration of marginal probabilities
             * @return                                    An unique pointer to an object of type
             *                                            `IJointProbabilityFunction` that has been created
             */
            virtual std::unique_ptr<IJointProbabilityFunction> create(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const = 0;

            /**
             * @see `IDistanceMeasureFactory::createDistanceMeasure`
             */
            std::unique_ptr<IDistanceMeasure> createDistanceMeasure(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const override final {
                return this->create(marginalProbabilityCalibrationModel, jointProbabilityCalibrationModel);
            }
    };

}
