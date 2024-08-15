#include "mlrl/seco/lift_functions/lift_function_peak.hpp"

#include "mlrl/common/data/array.hpp"
#include "mlrl/common/util/validation.hpp"

#include <algorithm>

namespace seco {

    static inline float64 calculateLiftInternally(uint32 numLabels, uint32 totalLabels, uint32 peakLabel,
                                                  float64 maxLift, float64 exponent) {
        if (numLabels == peakLabel) {
            return maxLift;
        } else {
            float64 normalization;

            if (numLabels < peakLabel) {
                normalization = ((float64) numLabels - 1) / ((float64) peakLabel - 1);
            } else {
                normalization =
                  ((float64) numLabels - (float64) totalLabels) / ((float64) totalLabels - (float64) peakLabel);
            }

            return 1 + pow(normalization, exponent) * (maxLift - 1);
        }
    }

    /**
     * A lift function that monotonously increases until a certain number of labels, where the maximum lift is reached,
     * and monotonously decreases afterwards.
     */
    class PeakLiftFunction final : public ILiftFunction {
        private:

            const uint32 numLabels_;

            const uint32 peakLabel_;

            const float64 maxLift_;

            const float64 exponent_;

            const Array<float64>& maxLiftsAfterPeak_;

        public:

            /**
             * @param numLabels         The total number of available labels. Must be greater than 0
             * @param peakLabel         The number of labels for which the lift is maximum. Must be in [1, numLabels]
             * @param maxLift           The lift at the peak label. Must be at least 1
             * @param curvature         The curvature of the lift function. A greater value results in a steeper curve,
             *                          a smaller value results in a flatter curve. Must be greater than 0
             * @param maxLiftsAfterPeak A reference to an object of type `Array<float64>` that specifies that maximum
                                        lifts that are possible by adding additional labels to heads that predict for
                                        more labels than the peak label
             */
            PeakLiftFunction(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature,
                             const Array<float64>& maxLiftsAfterPeak)
                : numLabels_(numLabels), peakLabel_(peakLabel), maxLift_(maxLift), exponent_(1.0 / curvature),
                  maxLiftsAfterPeak_(maxLiftsAfterPeak) {}

            float64 calculateLift(uint32 numLabels) const override {
                return calculateLiftInternally(numLabels, numLabels_, peakLabel_, maxLift_, exponent_);
            }

            float64 getMaxLift(uint32 numLabels) const override {
                if (numLabels < peakLabel_) {
                    return maxLift_;
                } else {
                    return maxLiftsAfterPeak_[numLabels - peakLabel_];
                }
            }
    };

    /**
     * Allows to create instances of the type `ILiftFunction` that monotonously increase until a certain number of
     * labels, where the maximum lift is reached, and monotonously decrease afterwards.
     */
    class PeakLiftFunctionFactory final : public ILiftFunctionFactory {
        private:

            const uint32 numLabels_;

            const uint32 peakLabel_;

            const float64 maxLift_;

            const float64 curvature_;

            Array<float64> maxLiftsAfterPeak_;

        public:

            /**
             * @param numLabels The total number of available labels. Must be greater than 0
             * @param peakLabel The index of the label for which the lift is maximal. Must be in [1, numLabels]
             * @param maxLift   The lift at the peak label. Must be at least 1
             * @param curvature The curvature of the lift function. A greater value results in a steeper curvature, a
             *                  smaller value results in a flatter curvature. Must be greater than 0
             */
            PeakLiftFunctionFactory(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature)
                : numLabels_(numLabels), peakLabel_(peakLabel), maxLift_(maxLift), curvature_(curvature),
                  maxLiftsAfterPeak_(numLabels - peakLabel) {
                for (uint32 i = 0; i < numLabels - peakLabel; i++) {
                    maxLiftsAfterPeak_[i] =
                      calculateLiftInternally(i + peakLabel, numLabels, peakLabel, maxLift, curvature);
                }
            }

            std::unique_ptr<ILiftFunction> create() const override {
                return std::make_unique<PeakLiftFunction>(numLabels_, peakLabel_, maxLift_, curvature_,
                                                          maxLiftsAfterPeak_);
            }
    };

    PeakLiftFunctionConfig::PeakLiftFunctionConfig() : peakLabel_(2), maxLift_(1.08), curvature_(1.0) {}

    uint32 PeakLiftFunctionConfig::getPeakLabel() const {
        return peakLabel_;
    }

    IPeakLiftFunctionConfig& PeakLiftFunctionConfig::setPeakLabel(uint32 peakLabel) {
        if (peakLabel != 0) util::assertGreaterOrEqual<uint32>("peakLabel", peakLabel, 1);
        peakLabel_ = peakLabel;
        return *this;
    }

    float64 PeakLiftFunctionConfig::getMaxLift() const {
        return maxLift_;
    }

    IPeakLiftFunctionConfig& PeakLiftFunctionConfig::setMaxLift(float64 maxLift) {
        util::assertGreaterOrEqual<float64>("maxLift", maxLift, 1);
        maxLift_ = maxLift;
        return *this;
    }

    float64 PeakLiftFunctionConfig::getCurvature() const {
        return curvature_;
    }

    IPeakLiftFunctionConfig& PeakLiftFunctionConfig::setCurvature(float64 curvature) {
        util::assertGreater<float64>("curvature", curvature, 0);
        curvature_ = curvature;
        return *this;
    }

    std::unique_ptr<ILiftFunctionFactory> PeakLiftFunctionConfig::createLiftFunctionFactory(
      const IRowWiseLabelMatrix& labelMatrix) const {
        uint32 numLabels = labelMatrix.getNumOutputs();
        uint32 peakLabel = peakLabel_;

        if (peakLabel > 0) {
            peakLabel = std::min(numLabels, peakLabel);
        } else {
            uint32 labelCardinality = static_cast<uint32>(std::round(labelMatrix.calculateLabelCardinality()));
            peakLabel = std::max<uint32>(labelCardinality, 1);
        }

        return std::make_unique<PeakLiftFunctionFactory>(numLabels, peakLabel, maxLift_, curvature_);
    }

}
