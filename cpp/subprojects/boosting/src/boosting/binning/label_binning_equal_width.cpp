#include "boosting/binning/label_binning_equal_width.hpp"
#include "boosting/data/statistic_vector_dense_label_wise.hpp"
#include "boosting/data/statistic_vector_dense_example_wise.hpp"
#include "common/math/math.hpp"
#include "common/binning/binning.hpp"
#include <limits>


namespace boosting {

    static inline constexpr float64 calculateStatistic(float64 gradient, float64 hessian,
                                                       float64 l2RegularizationWeight) {
        return divideOrZero<float64>(gradient, hessian + l2RegularizationWeight);
    }

    template<typename GradientIterator, typename HessianIterator>
    static inline void fetchLabelInfo(GradientIterator gradientsBegin, GradientIterator gradientsEnd,
                                      HessianIterator hessiansBegin, HessianIterator hessiansEnd,
                                      float64 l2RegularizationWeight, LabelInfo& labelInfo, float32 binRatio,
                                      uint32 minBins, uint32 maxBins) {
        uint32 numStatistics = gradientsEnd - gradientsBegin;

        if (numStatistics > 0) {
            // Find minimum and maximum among the positive gradients and negative gradients, respectively...
            uint32 numPositive = 0;
            uint32 numNegative = 0;
            labelInfo.minPositive = std::numeric_limits<float64>::infinity();
            labelInfo.maxPositive = 0;
            labelInfo.minNegative = 0;
            labelInfo.maxNegative = -std::numeric_limits<float64>::infinity();

            for (uint32 i = 0; i < numStatistics; i++) {
                float64 statistic = calculateStatistic(gradientsBegin[i], hessiansBegin[i], l2RegularizationWeight);

                if (statistic < 0) {
                    numNegative++;

                    if (statistic < labelInfo.minNegative) {
                        labelInfo.minNegative = statistic;
                    }

                    if (statistic > labelInfo.maxNegative) {
                        labelInfo.maxNegative = statistic;
                    }
                } else if (statistic > 0) {
                    numPositive++;

                    if (statistic < labelInfo.minPositive) {
                        labelInfo.minPositive = statistic;
                    }

                    if (statistic > labelInfo.maxPositive) {
                        labelInfo.maxPositive = statistic;
                    }
                }
            }

            labelInfo.numNegativeBins =
                numPositive > 0 ? calculateNumBins(numPositive, binRatio, minBins, maxBins) : 0;
            labelInfo.numPositiveBins =
                numNegative > 0 ? calculateNumBins(numNegative, binRatio, minBins, maxBins) : 0;
        } else {
            labelInfo.numPositiveBins = 0;
            labelInfo.numNegativeBins = 0;
        }
    }

    template<typename GradientIterator, typename HessianIterator>
    static inline void createBinsInternally(LabelInfo& labelInfo, GradientIterator gradientsBegin,
                                            GradientIterator gradientsEnd, HessianIterator hessiansBegin,
                                            HessianIterator hessiansEnd, float64 l2RegularizationWeight,
                                            ILabelBinning::Callback callback,
                                            ILabelBinning::ZeroCallback zeroCallback) {
        uint32 numPositiveBins = labelInfo.numPositiveBins;
        float64 minPositive = labelInfo.minPositive;
        float64 maxPositive = labelInfo.maxPositive;
        uint32 numNegativeBins = labelInfo.numNegativeBins;
        float64 minNegative = labelInfo.minNegative;
        float64 maxNegative = labelInfo.maxNegative;

        float64 spanPerNegativeBin = maxPositive > 0 ? (maxPositive - minPositive) / numNegativeBins : 0;
        float64 spanPerPositiveBin = minNegative < 0 ? (maxNegative - minNegative) / numPositiveBins : 0;

        // Assign labels to bins...
        uint32 numStatistics = gradientsEnd - gradientsBegin;

        for (uint32 i = 0; i < numStatistics; i++) {
            float64 gradient = gradientsBegin[i];
            float64 hessian = hessiansBegin[i];
            float64 statistic = calculateStatistic(gradient, hessian, l2RegularizationWeight);

            if (statistic > 0) {
                // Gradient is positive, i.e., label belongs to a negative bin...
                uint32 binIndex = std::floor((statistic - minPositive) / spanPerNegativeBin);

                if (binIndex >= numNegativeBins) {
                    binIndex = numNegativeBins - 1;
                }

                callback(binIndex, i, gradient, hessian);
            } else if (statistic < 0) {
                // Gradient is negative, i.e., label belongs to a positive bin...
                uint32 binIndex = std::floor((statistic - minNegative) / spanPerPositiveBin);

                if (binIndex >= numPositiveBins) {
                    binIndex = numPositiveBins - 1;
                }

                callback(numNegativeBins + binIndex, i, gradient, hessian);
            } else {
                zeroCallback(i);
            }
        }
    }

    EqualWidthLabelBinning::EqualWidthLabelBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
        : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

    }

    uint32 EqualWidthLabelBinning::getMaxBins(uint32 numLabels) const {
        return calculateNumBins(numLabels, binRatio_, minBins_, maxBins_) + 1;
    }

    LabelInfo EqualWidthLabelBinning::getLabelInfo(
            DenseLabelWiseStatisticVector::gradient_const_iterator gradientsBegin,
            DenseLabelWiseStatisticVector::gradient_const_iterator gradientsEnd,
            DenseLabelWiseStatisticVector::hessian_const_iterator hessiansBegin,
            DenseLabelWiseStatisticVector::hessian_const_iterator hessiansEnd, float64 l2RegularizationWeight) const {
        LabelInfo labelInfo;
        fetchLabelInfo(gradientsBegin, gradientsEnd, hessiansBegin, hessiansEnd, l2RegularizationWeight, labelInfo,
                       binRatio_, minBins_, maxBins_);
        return labelInfo;
    }

    LabelInfo EqualWidthLabelBinning::getLabelInfo(
            DenseExampleWiseStatisticVector::gradient_const_iterator gradientsBegin,
            DenseExampleWiseStatisticVector::gradient_const_iterator gradientsEnd,
            DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansBegin,
            DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansEnd,
            float64 l2RegularizationWeight) const {
        LabelInfo labelInfo;
        fetchLabelInfo(gradientsBegin, gradientsEnd, hessiansBegin, hessiansEnd, l2RegularizationWeight, labelInfo,
                       binRatio_, minBins_, maxBins_);
        return labelInfo;
    }

    void EqualWidthLabelBinning::createBins(LabelInfo labelInfo,
                                            DenseLabelWiseStatisticVector::gradient_const_iterator gradientsBegin,
                                            DenseLabelWiseStatisticVector::gradient_const_iterator gradientsEnd,
                                            DenseLabelWiseStatisticVector::hessian_const_iterator hessiansBegin,
                                            DenseLabelWiseStatisticVector::hessian_const_iterator hessiansEnd,
                                            float64 l2RegularizationWeight, ILabelBinning::Callback callback,
                                            ILabelBinning::ZeroCallback zeroCallback) const {
        createBinsInternally(labelInfo, gradientsBegin, gradientsEnd, hessiansBegin, hessiansEnd,
                             l2RegularizationWeight, callback, zeroCallback);
    }

    void EqualWidthLabelBinning::createBins(
            LabelInfo labelInfo, DenseExampleWiseStatisticVector::gradient_const_iterator gradientsBegin,
            DenseExampleWiseStatisticVector::gradient_const_iterator gradientsEnd,
            DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansBegin,
            DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansEnd,
            float64 l2RegularizationWeight, ILabelBinning::Callback callback,
            ILabelBinning::ZeroCallback zeroCallback) const {
        createBinsInternally(labelInfo, gradientsBegin, gradientsEnd, hessiansBegin, hessiansEnd,
                             l2RegularizationWeight, callback, zeroCallback);
    }

}
