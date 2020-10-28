#include "predictions.h"
#include "statistics.h"
#include <cstdlib>


AbstractPrediction::AbstractPrediction(uint32 numElements)
    : DenseVector<float64>(numElements) {

}

AbstractEvaluatedPrediction::AbstractEvaluatedPrediction(uint32 numElements)
    : AbstractPrediction(numElements) {

}

FullPrediction::FullPrediction(uint32 numElements)
    : AbstractEvaluatedPrediction(numElements), FullIndexVector(numElements) {

}

uint32 FullPrediction::getNumElements() const {
    return DenseVector<float64>::getNumElements();
}

void FullPrediction::setNumElements(uint32 numElements) {
    DenseVector<float64>::setNumElements(numElements);
    FullIndexVector::setNumElements(numElements);
}

void FullPrediction::apply(AbstractStatistics& statistics, uint32 statisticIndex) const {
    statistics.applyPrediction(statisticIndex, *this);
}

PartialPrediction::PartialPrediction(uint32 numElements)
    : AbstractEvaluatedPrediction(numElements), PartialIndexVector(numElements) {

}

uint32 PartialPrediction::getNumElements() const {
    return DenseVector<float64>::getNumElements();
}

void PartialPrediction::setNumElements(uint32 numElements) {
    DenseVector<float64>::setNumElements(numElements);
    PartialIndexVector::setNumElements(numElements);
}

void PartialPrediction::apply(AbstractStatistics& statistics, uint32 statisticIndex) const {
    statistics.applyPrediction(statisticIndex, *this);
}
