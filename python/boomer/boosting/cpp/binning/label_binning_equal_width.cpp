#include "label_binning_equal_width.h"
#include "../../../common/cpp/binning/binning_common.h"
#include "../data/vector_dense_label_wise.h"
#include "../data/vector_dense_example_wise.h"
#include <limits>

using namespace boosting;


template<class T>
EqualWidthLabelBinning<T>::EqualWidthLabelBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
    : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

}

template<class T>
uint32 EqualWidthLabelBinning<T>::getMaxBins(uint32 numLabels) const {
    return calculateNumBins(numLabels, binRatio_, minBins_, maxBins_) + 1;
}

template<class T>
LabelInfo EqualWidthLabelBinning<T>::getLabelInfo(const T& statisticVector) const {
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

        typename T::gradient_const_iterator iterator = statisticVector.gradients_cbegin();

        for (uint32 i = 0; i < numStatistics; i++) {
            float64 value = iterator[i];

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

        labelInfo.numNegativeBins = calculateNumBins(numPositive, binRatio_, minBins_, maxBins_);
        labelInfo.numPositiveBins = calculateNumBins(numNegative, binRatio_, minBins_, maxBins_);
    } else {
        labelInfo.numPositiveBins = 0;
        labelInfo.numNegativeBins = 0;
    }

    return labelInfo;
}

template<class T>
void EqualWidthLabelBinning<T>::createBins(LabelInfo labelInfo, const T& statisticVector,
                                           typename ILabelBinning<T>::Callback callback) const {
    uint32 numPositiveBins = labelInfo.numPositiveBins;
    float64 minPositive = labelInfo.minPositive;
    float64 maxPositive = labelInfo.maxPositive;
    uint32 numNegativeBins = labelInfo.numNegativeBins;
    float64 minNegative = labelInfo.minNegative;
    float64 maxNegative = labelInfo.minNegative;

    float64 spanPerPositiveBin = minNegative < 0 ? (maxNegative - minNegative) / numPositiveBins : 0;
    float64 spanPerNegativeBin = maxPositive > 0 ? (maxPositive - minPositive) / numNegativeBins : 0;

    // Assign labels to bins...
    uint32 numStatistics = statisticVector.getNumElements();
    typename T::gradient_const_iterator iterator = statisticVector.gradients_cbegin();

    for (uint32 i = 0; i < numStatistics; i++) {
        float64 value = iterator[i];

        if (value > 0) {
            // Gradient is positive, i.e., label belongs to a negative bin...
            uint32 binIndex = std::floor((value - minPositive) / spanPerNegativeBin);

            if (binIndex >= numNegativeBins) {
                binIndex = numNegativeBins - 1;
            }

            callback(binIndex, i, value);
        } else if (value < 0) {
            // Gradient is negative, i.e., label belongs to a positive bin...
            uint32 binIndex = std::floor((value - minNegative) / spanPerPositiveBin);

            if (binIndex >= numPositiveBins) {
                binIndex = numPositiveBins - 1;
            }

            callback(numNegativeBins + binIndex, i, value);
        }
    }
}

template class EqualWidthLabelBinning<DenseLabelWiseStatisticVector>;
template class EqualWidthLabelBinning<DenseExampleWiseStatisticVector>;
