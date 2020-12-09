#include "label_binning_equal_width.h"
#include "../data/vector_dense_label_wise.h"
#include "../data/vector_dense_example_wise.h"
#include <cmath>
#include <limits>

using namespace boosting;


template<class T>
LabelInfo EqualWidthLabelBinning<T>::getLabelInfo(const T& statisticVector, uint32 numPositiveBins,
                                                  uint32 numNegativeBins) const {
    LabelInfo labelInfo;
    uint32 numStatistics = statisticVector.getNumElements();

    if (numStatistics > 0) {
        // Find minimum and maximum among the positive gradients and negative gradients, respectively...
        uint32 numPositive = 0;
        uint32 numNegative = 0;
        labelInfo.minPositive = std::numeric_limits<float64>::infinity();
        labelInfo.maxPositive = 0;
        labelInfo.minNegative = 0;
        labelInfo.maxNegative = -std::numeric_limits<float64>::infinity();

        typename T::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();

        for (uint32 i = 0; i < numStatistics; i++) {
            float64 value = gradientIterator[i];

            if (value < 0) {
                numNegative++;

                if (value < labelInfo.minNegative) {
                    labelInfo.minNegative = value;
                }

                if (value > labelInfo.maxNegative) {
                    labelInfo.maxNegative = value;
                }
            } else if (value > 0) {
                numPositive++;

                if (value < labelInfo.minPositive) {
                    labelInfo.minPositive = value;
                }

                if (value > labelInfo.maxPositive) {
                    labelInfo.maxPositive = value;
                }
            }
        }

        labelInfo.numPositiveBins = numPositive > 0 ? numNegativeBins : 0;
        labelInfo.numNegativeBins = numNegative > 0 ? numPositiveBins : 0;
    } else {
        labelInfo.numPositiveBins = 0;
        labelInfo.numNegativeBins = 0;
    }

    return labelInfo;

}

template<class T>
void EqualWidthLabelBinning<T>::createBins(uint32 numPositiveBins, uint32 numNegativeBins, const T& statisticVector,
                                           IBinningObserver<float64>& observer) const {
    uint32 numGradients = statisticVector.getNumElements();
    typename T::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();

    // Find minimum and maximum among the positive gradients and negative gradients, respectively...
    float64 minPositiveGradient = std::numeric_limits<float64>::infinity();
    float64 maxPositiveGradient = 0;
    float64 minNegativeGradient = 0;
    float64 maxNegativeGradient = -std::numeric_limits<float64>::infinity();

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

        if (gradient > 0) {
            // Gradient is positive, i.e., label belongs to a negative bin...
            uint32 binIndex = std::floor((gradient - minPositiveGradient) / spanPerNegativeBin);

            if (binIndex >= numNegativeBins) {
                binIndex = numNegativeBins - 1;
            }

            observer.onBinUpdate(binIndex, i, gradient);
        } else if (gradient < 0) {
            // Gradient is negative, i.e., label belongs to a positive bin...
            uint32 binIndex = std::floor((gradient - minNegativeGradient) / spanPerPositiveBin);

            if (binIndex >= numPositiveBins) {
                binIndex = numPositiveBins - 1;
            }

            observer.onBinUpdate(numNegativeBins + binIndex, i, gradient);
        }
    }
}

template class EqualWidthLabelBinning<DenseLabelWiseStatisticVector>;
template class EqualWidthLabelBinning<DenseExampleWiseStatisticVector>;
