#pragma once

#include "common/data/vector_dense.hpp"
#include "common/iterator/binary_forward_iterator.hpp"
#include "common/math/math.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * Calculates and returns the joint probability of a specific label vector being the true label vector of an
     * example, based on scores that are predicted for each label.
     *
     * @param scoresIterator                An iterator of type `VectorConstView<float64>::const_iterator` that provides
     *                                      access to the scores predicted for individual labels
     * @param numLabels                     The total number of available labels
     * @param labelVector                   A reference to an object of type `LabelVector` representing the label vector
     *                                      for which the joint probability should be calculated
     * @param labelWiseProbabilityFunction  A reference to an object of type `ILabelWiseProbabilityFunction` that should
     *                                      be used to transform the regression scores that are predicted for individual
     *                                      labels into marginal probabilities
     * @return                              The joint probability that has been calculated
     */
    static inline float64 calculateJointProbability(VectorConstView<float64>::const_iterator scoreIterator,
                                                    uint32 numLabels, const LabelVector& labelVector,
                                                    const ILabelWiseProbabilityFunction& labelWiseProbabilityFunction) {
        auto labelIterator = make_binary_forward_iterator(labelVector.cbegin(), labelVector.cend());
        float64 jointProbability = 1;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 score = scoreIterator[i];
            float64 probability = labelWiseProbabilityFunction.transformScoreIntoProbability(score);
            bool trueLabel = *labelIterator;

            if (!trueLabel) {
                probability = 1 - probability;
            }

            jointProbability *= probability;
            labelIterator++;
        }

        return jointProbability;
    }

    /**
     * Calculates and returns the joint probability of all label vectors in a `LabelVectorSet` being the true label
     * vector of an example, based on scores that are predicted for each label.
     *
     * @param scoreIterator                 An iterator of type `VectorConstView<float64>::const_iterator` that provides
     *                                      access to the scores predicted for individual labels
     * @param numLabels                     The total number of available labels
     * @param labelVectorSet                A reference to an object of type `LabelVectorSet` that stores the label
     *                                      vectors for which joint probabilities should be calculated
     * @param labelWiseProbabilityFunction  A reference to an object of type `ILabelWiseProbabilityFunction` that should
     *                                      be used to transform the regression scores that are predicted for individual
     *                                      labels into marginal probabilities
     * @return                              A `std::pair` that stores an unique pointer to a `DenseVector` storing the
     *                                      joint probabilities that have been calculated, as well as the sum of these
     *                                      probabilities
     */
    static inline std::pair<std::unique_ptr<DenseVector<float64>>, float64> calculateJointProbabilities(
      VectorConstView<float64>::const_iterator scoreIterator, uint32 numLabels, const LabelVectorSet& labelVectorSet,
      const ILabelWiseProbabilityFunction& labelWiseProbabilityFunction) {
        uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();
        std::unique_ptr<DenseVector<float64>> jointProbabilityVectorPtr =
          std::make_unique<DenseVector<float64>>(numLabelVectors);
        DenseVector<float64>::iterator jointProbabilityIterator = jointProbabilityVectorPtr->begin();
        LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet.cbegin();
        const LabelVector& firstLabelVector = *((*labelVectorIterator).first);
        float64 sumOfJointProbabilities =
          calculateJointProbability(scoreIterator, numLabels, firstLabelVector, labelWiseProbabilityFunction);
        jointProbabilityIterator[0] = sumOfJointProbabilities;
        labelVectorIterator++;
        uint32 i = 1;

        for (; labelVectorIterator != labelVectorSet.cend(); labelVectorIterator++) {
            const LabelVector& labelVector = *((*labelVectorIterator).first);
            float64 jointProbability =
              calculateJointProbability(scoreIterator, numLabels, labelVector, labelWiseProbabilityFunction);
            sumOfJointProbabilities += jointProbability;
            jointProbabilityIterator[i] = jointProbability;
            i++;
        }

        return std::make_pair(std::move(jointProbabilityVectorPtr), sumOfJointProbabilities);
    }

    /**
     * Normalizes and returns a joint probability, which was previously calculated via the function
     * `calculateJointProbabilities`.
     *
     * @param jointProbability          The joint probability to be normalized
     * @param sumOfJointProbabilities   The sum of all joint probabilities
     * @return                          The normalized joint probability
     */
    static inline constexpr float64 normalizeJointProbability(float64 jointProbability,
                                                              float64 sumOfJointProbabilities) {
        return divideOrZero(jointProbability, sumOfJointProbabilities);
    }

}
