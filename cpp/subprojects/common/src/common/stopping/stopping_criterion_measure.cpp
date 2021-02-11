#include "common/stopping/stopping_criterion_measure.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/math/math.hpp"
#include <iostream>


static inline float64 evaluateOnHoldoutSet(const BiPartition& partition, const IStatistics& statistics,
                                           const IEvaluationMeasure& measure) {
    uint32 numHoldoutExamples = partition.getNumSecond();
    BiPartition::const_iterator iterator = partition.second_cbegin();
    float64 mean = 0;

    for (uint32 i = 0; i < numHoldoutExamples; i++) {
        uint32 exampleIndex = iterator[i];
        float64 score = statistics.evaluatePrediction(exampleIndex, measure);
        mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
    }

    return mean;
}

float64 MinFunction::aggregate(uint32 numElements, RingBuffer<float64>::const_iterator iterator) const {
    float64 min = iterator[0];

    for (uint32 i = 1; i < numElements; i++) {
        float64 value = iterator[i];

        if (value < min) {
            min = value;
        }
    }

    return min;
}

float64 MaxFunction::aggregate(uint32 numElements, RingBuffer<float64>::const_iterator iterator) const {
    float64 max = iterator[0];

    for (uint32 i = 1; i < numElements; i++) {
        float64 value = iterator[i];

        if (value > max) {
            max = value;
        }
    }

    return max;
}

float64 ArithmeticMeanFunction::aggregate(uint32 numElements, RingBuffer<float64>::const_iterator iterator) const {
    float64 mean = 0;

    for (uint32 i = 0; i < numElements; i++) {
        float64 value = iterator[i];
        mean = iterativeArithmeticMean<float64>(i + 1, value, mean);
    }

    return mean;
}

MeasureStoppingCriterion::MeasureStoppingCriterion(std::shared_ptr<IEvaluationMeasure> measurePtr,
                                                   std::shared_ptr<IAggregationFunction> aggregationFunctionPtr,
                                                   uint32 minRules, uint32 updateInterval, uint32 stopInterval,
                                                   uint32 bufferSize, float64 minImprovement)
    : measurePtr_(measurePtr), aggregationFunctionPtr_(aggregationFunctionPtr), minRules_(minRules),
      updateInterval_(updateInterval), stopInterval_(stopInterval), minImprovement_(minImprovement),
      buffer_(RingBuffer<float64>(bufferSize)) {
    uint32 bufferInterval = bufferSize * updateInterval;
    offset_ = bufferInterval < minRules ? minRules - bufferInterval : 0;
}

bool MeasureStoppingCriterion::shouldContinue(const IPartition& partition, const IStatistics& statistics,
                                              uint32 numRules) {
    bool result = true;

    if (numRules > offset_ && numRules % updateInterval_ == 0) {
        const BiPartition& biPartition = static_cast<const BiPartition&>(partition);
        float64 currentScore = evaluateOnHoldoutSet(biPartition, statistics, *measurePtr_);

        if (numRules >= minRules_ && numRules % stopInterval_ == 0) {
            uint32 numBufferedElements = buffer_.getNumElements();

            if (numBufferedElements > 0) {
                float64 aggregatedScore = aggregationFunctionPtr_->aggregate(numBufferedElements, buffer_.cbegin());
                float64 percentageImprovement = (aggregatedScore - currentScore) / currentScore;
                result = percentageImprovement > minImprovement_;
                std::cout << numRules << ": improvement = " << percentageImprovement << " ==> " << (result ? "continue" : "stop") << "\n";
            }
        }

        buffer_.push(currentScore);
    }

    return result;
}
