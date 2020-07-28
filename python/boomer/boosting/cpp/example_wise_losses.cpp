#include "example_wise_losses.h"

using namespace losses;


AbstractExampleWiseLoss::~AbstractExampleWiseLoss() {

}

void AbstractExampleWiseLoss::calculateGradientsAndHessians(intp exampleIndex, float64* predictedScores,
                                                            float64* gradients, float64* hessians) {

}
