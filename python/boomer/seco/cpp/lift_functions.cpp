#include "lift_functions.h"
#include <math.h>

using namespace seco;


AbstractLiftFunction::~AbstractLiftFunction() {

}

float64 AbstractLiftFunction::calculateLift(uint32 numLabels) {
    return 0;
}

float64 AbstractLiftFunction::getMaxLift() {
    return 0;
}

PeakLiftFunctionImpl::PeakLiftFunctionImpl(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature) {
    numLabels_ = numLabels;
    peakLabel_ = peakLabel;
    maxLift_ = maxLift;
    exponent_ = 1.0 / curvature;
}

PeakLiftFunctionImpl::~PeakLiftFunctionImpl() {

}

float64 PeakLiftFunctionImpl::calculateLift(uint32 numLabels) {
    float64 normalization;

    if (numLabels < peakLabel_) {
        normalization = ((float64) numLabels - 1) / ((float64) peakLabel_ - 1);
    } else if (numLabels > peakLabel_) {
        normalization = ((float64) numLabels - (float64) numLabels_) / ((float64) numLabels_ - (float64) peakLabel_);
    } else {
        return maxLift_;
    }

    return 1 + pow(normalization, exponent_) * (maxLift_ - 1);
}

float64 PeakLiftFunctionImpl::getMaxLift() {
    return maxLift_;
}
