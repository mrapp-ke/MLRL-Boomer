#include "boosting/binning/label_binning_equal_width.hpp"
#include "boosting/data/statistic_vector_dense_label_wise.hpp"
#include "boosting/data/statistic_vector_dense_example_wise.hpp"
#include "common/math/math.hpp"
#include "common/binning/binning.hpp"
#include "common/validation.hpp"
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

    /**
     * Assigns labels to bins, based on the corresponding gradients and Hessians, in a way such that each bin contains
     * labels for which the predicted score is expected to belong to the same value range.
     */
    class EqualWidthLabelBinning final : public ILabelBinning {

        private:

            float32 binRatio_;

            uint32 minBins_;

            uint32 maxBins_;

        public:

            /**
             * @param binRatio  A percentage that specifies how many bins should be used to assign labels to, e.g., if
             *                  100 labels are available, 0.5 means that `ceil(0.5 * 100) = 50` bins should be used.
             *                  Must be in (0, 1)
             * @param minBins   The minimum number of bins to be used to assign labels to. Must be at least 2
             * @param maxBins   The maximum number of bins to be used to assign labels to. Must be at least `minBins` or
             *                  0, if the maximum number of bins should not be restricted
             */
            EqualWidthLabelBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
                : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {
                assertGreater<float32>("binRatio", binRatio, 0.0);
                assertSmaller<uint32>("binRatio", binRatio, 1.0);
                assertGreaterOrEqual<uint32>("minBins", minBins, 2);
                if (maxBins != 0) { assertGreaterOrEqual<uint32>("maxBins", maxBins, minBins); }
            }

            uint32 getMaxBins(uint32 numLabels) const override {
                return calculateNumBins(numLabels, binRatio_, minBins_, maxBins_) + 1;
            }

            LabelInfo getLabelInfo(DenseLabelWiseStatisticVector::gradient_const_iterator gradientsBegin,
                                   DenseLabelWiseStatisticVector::gradient_const_iterator gradientsEnd,
                                   DenseLabelWiseStatisticVector::hessian_const_iterator hessiansBegin,
                                   DenseLabelWiseStatisticVector::hessian_const_iterator hessiansEnd,
                                   float64 l2RegularizationWeight) const override {
                LabelInfo labelInfo;
                fetchLabelInfo(gradientsBegin, gradientsEnd, hessiansBegin, hessiansEnd, l2RegularizationWeight,
                               labelInfo, binRatio_, minBins_, maxBins_);
                return labelInfo;
            }

            LabelInfo getLabelInfo(DenseExampleWiseStatisticVector::gradient_const_iterator gradientsBegin,
                                   DenseExampleWiseStatisticVector::gradient_const_iterator gradientsEnd,
                                   DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansBegin,
                                   DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansEnd,
                                   float64 l2RegularizationWeight) const override {
                LabelInfo labelInfo;
                fetchLabelInfo(gradientsBegin, gradientsEnd, hessiansBegin, hessiansEnd, l2RegularizationWeight,
                               labelInfo, binRatio_, minBins_, maxBins_);
                return labelInfo;
            }

            void createBins(LabelInfo labelInfo, DenseLabelWiseStatisticVector::gradient_const_iterator gradientsBegin,
                            DenseLabelWiseStatisticVector::gradient_const_iterator gradientsEnd,
                            DenseLabelWiseStatisticVector::hessian_const_iterator hessiansBegin,
                            DenseLabelWiseStatisticVector::hessian_const_iterator hessiansEnd,
                            float64 l2RegularizationWeight, Callback callback,
                            ZeroCallback zeroCallback) const override {
                createBinsInternally(labelInfo, gradientsBegin, gradientsEnd, hessiansBegin, hessiansEnd,
                                     l2RegularizationWeight, callback, zeroCallback);
            }

            void createBins(LabelInfo labelInfo,
                            DenseExampleWiseStatisticVector::gradient_const_iterator gradientsBegin,
                            DenseExampleWiseStatisticVector::gradient_const_iterator gradientsEnd,
                            DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansBegin,
                            DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansEnd,
                            float64 l2RegularizationWeight, Callback callback,
                            ZeroCallback zeroCallback) const override {
                createBinsInternally(labelInfo, gradientsBegin, gradientsEnd, hessiansBegin, hessiansEnd,
                                     l2RegularizationWeight, callback, zeroCallback);
            }

    };

    EqualWidthLabelBinningFactory::EqualWidthLabelBinningFactory(float32 binRatio, uint32 minBins, uint32 maxBins)
        : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

    }

    std::unique_ptr<ILabelBinning> EqualWidthLabelBinningFactory::create() const {
        return std::make_unique<EqualWidthLabelBinning>(binRatio_, minBins_, maxBins_);
    }

}
