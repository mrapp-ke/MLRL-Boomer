#include "lift_functions.h"
#include <math.h>

using namespace seco;


AbstractLiftFunction::~AbstractLiftFunction() {

}

float64 AbstractLiftFunction::calculateLift(intp numLabels) {
    return 0;
}

float64 AbstractLiftFunction::getMaxLift() {
    return 0;
}

PeakLiftFunctionImpl::PeakLiftFunctionImpl(intp numLabels, intp peakLabel, float64 maxLift, float64 curvature) {
    numLabels_ = numLabels;
    peakLabel_ = peakLabel;
    maxLift_ = maxLift;
    curvature_ = curvature;
}

PeakLiftFunctionImpl::~PeakLiftFunctionImpl() {

}

float64 PeakLiftFunctionImpl::calculateLift(intp numLabels) {
    float64 normalization;

    if (numLabels < peakLabel_) {
        normalization = (numLabels - 1) / ((float64) (peakLabel_ - 1));
    } else if (numLabels > peakLabel_) {
        normalization = (numLabels - numLabels_) / ((float64) (numLabels - peakLabel_));
    } else {
        return maxLift_;
    }

    return 1 + pow(normalization, curvature_) * (maxLift_ - 1);
}

float64 PeakLiftFunctionImpl::getMaxLift() {
    return maxLift_;
}
