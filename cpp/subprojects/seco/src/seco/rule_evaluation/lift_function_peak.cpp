#include "seco/rule_evaluation/lift_function_peak.hpp"
#include "common/util/validation.hpp"
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

    PeakLiftFunctionConfig::PeakLiftFunctionConfig()
        : numLabels_(10), peakLabel_(2), maxLift_(1.08), curvature_(1.0) {

    }

    uint32 PeakLiftFunctionConfig::getNumLabels() const {
        return numLabels_;
    }

    PeakLiftFunctionConfig& PeakLiftFunctionConfig::setNumLabels(uint32 numLabels) {
        assertGreater<uint32>("numLabels", numLabels, 0);
        numLabels_ = numLabels;
        return *this;
    }

    uint32 PeakLiftFunctionConfig::getPeakLabel() const {
        return peakLabel_;
    }

    PeakLiftFunctionConfig& PeakLiftFunctionConfig::setPeakLabel(uint32 peakLabel) {
        assertGreaterOrEqual<uint32>("peakLabel", peakLabel, 0);
        peakLabel_ = peakLabel;
        return *this;
    }

    float64 PeakLiftFunctionConfig::getMaxLift() const {
        return maxLift_;
    }

    PeakLiftFunctionConfig& PeakLiftFunctionConfig::setMaxLift(float64 maxLift) {
        assertGreaterOrEqual<float64>("maxLift", maxLift, 1);
        maxLift_ = maxLift;
        return *this;
    }

    float64 PeakLiftFunctionConfig::getCurvature() const {
        return curvature_;
    }

    PeakLiftFunctionConfig& PeakLiftFunctionConfig::setCurvature(float64 curvature) {
        assertGreater<float64>("curvature", curvature, 0);
        curvature_ = curvature;
        return *this;
    }

    PeakLiftFunctionFactory::PeakLiftFunctionFactory(uint32 numLabels, uint32 peakLabel, float64 maxLift,
                                                     float64 curvature)
        : numLabels_(numLabels), peakLabel_(peakLabel), maxLift_(maxLift), curvature_(curvature) {

    }

    std::unique_ptr<ILiftFunction> PeakLiftFunctionFactory::create() const {
        return std::make_unique<PeakLiftFunction>(numLabels_, peakLabel_, maxLift_, curvature_);
    }

}
