#include "label_wise_losses.h"

using namespace losses;


AbstractLabelWiseLoss::~AbstractLabelWiseLoss() {

}

std::pair<float64, float64> AbstractLabelWiseLoss::calculateGradientAndHessian(
        statistics::AbstractLabelMatrix* labelMatrix,  intp exampleIndex, intp labelIndex, float64 predictedScore) {
    return std::make_pair(0, 0);
}
