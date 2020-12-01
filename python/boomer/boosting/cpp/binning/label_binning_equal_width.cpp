#include "label_binning_equal_width.h"
#include <cmath>
#include <limits>


template<class T>
void EqualWidthLabelBinning<T>::createBins(uint32 numPositiveBins, uint32 numNegativeBins, const T& statisticVector,
                                           IBinningObserver<float64>& observer) const {
    uint32 numGradients = statisticVector.getNumElements();
    typename T::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();

    // Find minimum and maximum among the positive gradients and negative gradients, respectively...
    float64 minPositiveGradient = std::numeric_limits<float64>::max();
    float64 maxPositiveGradient = 0;
    float64 minNegativeGradient = 0;
    float64 maxNegativeGradient = std::numeric_limits<float64>::min();

    for (uint32 i = 0; i < numGradients; i++) {
        float64 gradient = gradientIterator[i];

        if (gradient < 0) {
            // Gradient is negative...
            if (gradient < minNegativeGradient) {
                minNegativeGradient = gradient;
            }

            if (gradient > maxNegativeGradient) {
                maxNegativeGradient = gradient;
            }
        } else if (gradient > 0) {
            // Gradient is positive...
            if (gradient < minPositiveGradient) {
                minPositiveGradient = gradient;
            }

            if (gradient > maxPositiveGradient) {
                maxPositiveGradient = gradient;
            }
        }
    }

    float64 spanPerPositiveBin =
        minNegativeGradient < 0 ? (maxNegativeGradient - minNegativeGradient) / numPositiveBins : 0;
    float64 spanPerNegativeBin =
        maxPositiveGradient > 0 ? (maxPositiveGradient - minPositiveGradient) / numNegativeBins : 0;

    // Assign labels to bins...
    for (uint32 i = 0; i < numGradients; i++) {
        float64 gradient = gradientIterator[i];
        uint32 binIndex;

        if (gradient > 0) {
            // Gradient is positive, i.e., label belongs to a negative bin...
            binIndex = std::floor((gradient - minPositiveGradient) / spanPerNegativeBin);

            if (binIndex >= numNegativeBins) {
                binIndex = numNegativeBins - 1;
            }
        } else if (gradient < 0) {
            // Gradient is negative, i.e., label belongs to a positive bin...
            binIndex = std::floor((gradient - minNegativeGradient) / spanPerPositiveBin);

            if (binIndex >= numPositiveBins) {
                binIndex = numPositiveBins - 1;
            }

            binIndex += numNegativeBins;
        }

        observer.onBinUpdate(binIndex, i, gradient);
    }
}
