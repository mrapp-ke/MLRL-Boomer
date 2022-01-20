#include "seco/lift_functions/lift_function_peak.hpp"
#include "common/util/validation.hpp"
#include <algorithm>
#include <cmath>


namespace seco {

    /**
     * A lift function that monotonously increases until a certain number of labels, where the maximum lift is reached,
     * and monotonously decreases afterwards.
     */
    class PeakLiftFunction final : public ILiftFunction {

        private:

            uint32 numLabels_;

            uint32 peakLabel_;

            float64 maxLift_;

            float64 exponent_;

        public:

            /**
             * @param numLabels The total number of available labels. Must be greater than 0
             * @param peakLabel The number of labels for which the lift is maximum. Must be in [1, numLabels]
             * @param maxLift   The lift at the peak label. Must be at least 1
             * @param curvature The curvature of the lift function. A greater value results in a steeper curvature, a
             *                  smaller value results in a flatter curvature. Must be greater than 0
             */
            PeakLiftFunction(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature)
                : numLabels_(numLabels), peakLabel_(peakLabel), maxLift_(maxLift), exponent_(1.0 / curvature) {

            }

            float64 calculateLift(uint32 numLabels) const override {
                float64 normalization;

                if (numLabels < peakLabel_) {
                    normalization = ((float64) numLabels - 1) / ((float64) peakLabel_ - 1);
                } else if (numLabels > peakLabel_) {
                    normalization = ((float64) numLabels - (float64) numLabels_)
                                    / ((float64) numLabels_ - (float64) peakLabel_);
                } else {
                    return maxLift_;
                }

                return 1 + pow(normalization, exponent_) * (maxLift_ - 1);
            }

            float64 getMaxLift() const override {
                return maxLift_;
            }

    };

    /**
     * Allows to create instances of the type `ILiftFunction` that monotonously increase until a certain number of
     * labels, where the maximum lift is reached, and monotonously decrease afterwards.
     */
    class PeakLiftFunctionFactory final : public ILiftFunctionFactory {

        private:

            uint32 numLabels_;

            uint32 peakLabel_;

            float64 maxLift_;

            float64 curvature_;

        public:

            /**
             * @param numLabels The total number of available labels. Must be greater than 0
             * @param peakLabel The index of the label for which the lift is maximal. Must be in [1, numLabels]
             * @param maxLift   The lift at the peak label. Must be at least 1
             * @param curvature The curvature of the lift function. A greater value results in a steeper curvature, a
             *                  smaller value results in a flatter curvature. Must be greater than 0
             */
            PeakLiftFunctionFactory(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature)
                : numLabels_(numLabels), peakLabel_(peakLabel), maxLift_(maxLift), curvature_(curvature) {

            }

            std::unique_ptr<ILiftFunction> create() const override {
                return std::make_unique<PeakLiftFunction>(numLabels_, peakLabel_, maxLift_, curvature_);
            }

    };

    PeakLiftFunctionConfig::PeakLiftFunctionConfig()
        : peakLabel_(2), maxLift_(1.08), curvature_(1.0) {

    }

    uint32 PeakLiftFunctionConfig::getPeakLabel() const {
        return peakLabel_;
    }

    IPeakLiftFunctionConfig& PeakLiftFunctionConfig::setPeakLabel(uint32 peakLabel) {
        if (peakLabel != 0) { assertGreaterOrEqual<uint32>("peakLabel", peakLabel, 1); }
        peakLabel_ = peakLabel;
        return *this;
    }

    float64 PeakLiftFunctionConfig::getMaxLift() const {
        return maxLift_;
    }

    IPeakLiftFunctionConfig& PeakLiftFunctionConfig::setMaxLift(float64 maxLift) {
        assertGreaterOrEqual<float64>("maxLift", maxLift, 1);
        maxLift_ = maxLift;
        return *this;
    }

    float64 PeakLiftFunctionConfig::getCurvature() const {
        return curvature_;
    }

    IPeakLiftFunctionConfig& PeakLiftFunctionConfig::setCurvature(float64 curvature) {
        assertGreater<float64>("curvature", curvature, 0);
        curvature_ = curvature;
        return *this;
    }

    std::unique_ptr<ILiftFunctionFactory> PeakLiftFunctionConfig::configure(
            const IRowWiseLabelMatrix& labelMatrix) const {
        uint32 numLabels = labelMatrix.getNumRows();
        uint32 peakLabel = peakLabel_ > 0 ? std::min(numLabels, peakLabel_)
                                          : std::max<uint32>(std::round(labelMatrix.calculateLabelCardinality()), 1);
        return std::make_unique<PeakLiftFunctionFactory>(numLabels, peakLabel, maxLift_, curvature_);
    }

}
