#include "mlrl/boosting/binning/label_binning_equal_width.hpp"

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete_binned.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_dynamic_binned.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_fixed_binned.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_complete_binned.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_dynamic_binned.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_fixed_binned.hpp"
#include "mlrl/common/util/math.hpp"
#include "mlrl/common/util/validation.hpp"

namespace boosting {

    /**
     * Assigns labels to bins, based on the corresponding criteria, in a way such that each bin contains labels for
     * which the predicted score is expected to belong to the same value range.
     */
    template<typename CriteriaType>
    class EqualWidthLabelBinning final : public ILabelBinning<CriteriaType> {
        private:

            const float32 binRatio_;

            const uint32 minBins_;

            const uint32 maxBins_;

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
                : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {}

            uint32 getMaxBins(uint32 numLabels) const override {
                return util::calculateBoundedFraction(numLabels, binRatio_, minBins_, maxBins_) + 1;
            }

            LabelInfo<CriteriaType> getLabelInfo(typename View<CriteriaType>::const_iterator criteria,
                                                 uint32 numCriteria) const override {
                LabelInfo<CriteriaType> labelInfo;
                labelInfo.numNegativeBins = 0;
                labelInfo.numPositiveBins = 0;

                if (numCriteria > 0) {
                    labelInfo.minNegative = 0;
                    labelInfo.maxNegative = -std::numeric_limits<CriteriaType>::infinity();
                    labelInfo.minPositive = std::numeric_limits<CriteriaType>::infinity();
                    labelInfo.maxPositive = 0;

                    for (uint32 i = 0; i < numCriteria; i++) {
                        CriteriaType criterion = criteria[i];

                        if (criterion < 0) {
                            labelInfo.numNegativeBins++;

                            if (criterion < labelInfo.minNegative) {
                                labelInfo.minNegative = criterion;
                            }

                            if (criterion > labelInfo.maxNegative) {
                                labelInfo.maxNegative = criterion;
                            }
                        } else if (criterion > 0) {
                            labelInfo.numPositiveBins++;

                            if (criterion < labelInfo.minPositive) {
                                labelInfo.minPositive = criterion;
                            }

                            if (criterion > labelInfo.maxPositive) {
                                labelInfo.maxPositive = criterion;
                            }
                        }
                    }

                    if (labelInfo.numNegativeBins > 0) {
                        labelInfo.numNegativeBins =
                          util::calculateBoundedFraction(labelInfo.numNegativeBins, binRatio_, minBins_, maxBins_);
                    }

                    if (labelInfo.numPositiveBins > 0) {
                        labelInfo.numPositiveBins =
                          util::calculateBoundedFraction(labelInfo.numPositiveBins, binRatio_, minBins_, maxBins_);
                    }
                }

                return labelInfo;
            }

            void createBins(LabelInfo<CriteriaType> labelInfo, typename View<CriteriaType>::const_iterator criteria,
                            uint32 numCriteria, typename ILabelBinning<CriteriaType>::Callback callback,
                            typename ILabelBinning<CriteriaType>::ZeroCallback zeroCallback) const override {
                uint32 numNegativeBins = labelInfo.numNegativeBins;
                CriteriaType minNegative = labelInfo.minNegative;
                CriteriaType maxNegative = labelInfo.maxNegative;
                uint32 numPositiveBins = labelInfo.numPositiveBins;
                CriteriaType minPositive = labelInfo.minPositive;
                CriteriaType maxPositive = labelInfo.maxPositive;

                CriteriaType spanPerNegativeBin = minNegative < 0 ? (maxNegative - minNegative) / numNegativeBins : 0;
                CriteriaType spanPerPositiveBin = maxPositive > 0 ? (maxPositive - minPositive) / numPositiveBins : 0;

                for (uint32 i = 0; i < numCriteria; i++) {
                    CriteriaType criterion = criteria[i];

                    if (criterion < 0) {
                        uint32 binIndex =
                          static_cast<uint32>(std::floor((criterion - minNegative) / spanPerNegativeBin));

                        if (binIndex >= numNegativeBins) {
                            binIndex = numNegativeBins - 1;
                        }

                        callback(binIndex, i);
                    } else if (criterion > 0) {
                        uint32 binIndex =
                          static_cast<uint32>(std::floor((criterion - minPositive) / spanPerPositiveBin));

                        if (binIndex >= numPositiveBins) {
                            binIndex = numPositiveBins - 1;
                        }

                        callback(numNegativeBins + binIndex, i);
                    } else {
                        zeroCallback(i);
                    }
                }
            }
    };

    /**
     * Allows to create instances of the class `EqualWidthLabelBinning` that assign labels to bins in a way such that
     * each bin contains labels for which the predicted score is expected to belong to the same value range.
     */
    class EqualWidthLabelBinningFactory final : public ILabelBinningFactory {
        private:

            const float32 binRatio_;

            const uint32 minBins_;

            const uint32 maxBins_;

        public:

            /**
             * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 labels are a
             *                  available, a percentage of 0.5 means that `ceil(0.5 * 100) = 50` bins should be used.
             *                  Must be in (0, 1)
             * @param minBins   The minimum number of bins that should be used. Must be at least 2
             * @param maxBins   The maximum number of bins that should be used. Must be at least `minBins` or 0, if the
             *                  maximum number of bins should not be restricted
             */
            EqualWidthLabelBinningFactory(float32 binRatio, uint32 minBins, uint32 maxBins)
                : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {}

            std::unique_ptr<ILabelBinning<float32>> create32Bit() const override {
                return std::make_unique<EqualWidthLabelBinning<float32>>(binRatio_, minBins_, maxBins_);
            }

            std::unique_ptr<ILabelBinning<float64>> create64Bit() const override {
                return std::make_unique<EqualWidthLabelBinning<float64>>(binRatio_, minBins_, maxBins_);
            }
    };

    EqualWidthLabelBinningConfig::EqualWidthLabelBinningConfig(
      ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
      ReadableProperty<IRegularizationConfig> l2RegularizationConfig)
        : binRatio_(0.04f), minBins_(1), maxBins_(0), l1RegularizationConfig_(l1RegularizationConfig),
          l2RegularizationConfig_(l2RegularizationConfig) {}

    float32 EqualWidthLabelBinningConfig::getBinRatio() const {
        return binRatio_;
    }

    IEqualWidthLabelBinningConfig& EqualWidthLabelBinningConfig::setBinRatio(float32 binRatio) {
        util::assertGreater<float32>("binRatio", binRatio, 0);
        util::assertLess<float32>("binRatio", binRatio, 1);
        binRatio_ = binRatio;
        return *this;
    }

    uint32 EqualWidthLabelBinningConfig::getMinBins() const {
        return minBins_;
    }

    IEqualWidthLabelBinningConfig& EqualWidthLabelBinningConfig::setMinBins(uint32 minBins) {
        util::assertGreaterOrEqual<uint32>("minBins", minBins, 1);
        minBins_ = minBins;
        return *this;
    }

    uint32 EqualWidthLabelBinningConfig::getMaxBins() const {
        return maxBins_;
    }

    IEqualWidthLabelBinningConfig& EqualWidthLabelBinningConfig::setMaxBins(uint32 maxBins) {
        if (maxBins != 0) util::assertGreaterOrEqual<uint32>("maxBins", maxBins, minBins_);
        maxBins_ = maxBins;
        return *this;
    }

    std::unique_ptr<IDecomposableRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createDecomposableCompleteRuleEvaluationFactory() const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluationFactory>(
          l1RegularizationWeight, l2RegularizationWeight, std::move(labelBinningFactoryPtr));
    }

    std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio,
                                                                                        uint32 minOutputs,
                                                                                        uint32 maxOutputs) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<DecomposableFixedPartialBinnedRuleEvaluationFactory>(
          outputRatio, minOutputs, maxOutputs, l1RegularizationWeight, l2RegularizationWeight,
          std::move(labelBinningFactoryPtr));
    }

    std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createDecomposableDynamicPartialRuleEvaluationFactory(float32 threshold,
                                                                                          float32 exponent) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluationFactory>(
          threshold, exponent, l1RegularizationWeight, l2RegularizationWeight, std::move(labelBinningFactoryPtr));
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createNonDecomposableCompleteRuleEvaluationFactory(
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<NonDecomposableCompleteBinnedRuleEvaluationFactory>(
          l1RegularizationWeight, l2RegularizationWeight, std::move(labelBinningFactoryPtr), blasFactory,
          lapackFactory);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createNonDecomposableFixedPartialRuleEvaluationFactory(
        float32 outputRatio, uint32 minOutputs, uint32 maxOutputs, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<NonDecomposableFixedPartialBinnedRuleEvaluationFactory>(
          outputRatio, minOutputs, maxOutputs, l1RegularizationWeight, l2RegularizationWeight,
          std::move(labelBinningFactoryPtr), blasFactory, lapackFactory);
    }

    std::unique_ptr<INonDecomposableRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createNonDecomposableDynamicPartialRuleEvaluationFactory(
        float32 threshold, float32 exponent, const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<NonDecomposableDynamicPartialBinnedRuleEvaluationFactory>(
          threshold, exponent, l1RegularizationWeight, l2RegularizationWeight, std::move(labelBinningFactoryPtr),
          blasFactory, lapackFactory);
    }

}
