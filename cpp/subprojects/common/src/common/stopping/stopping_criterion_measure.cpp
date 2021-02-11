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
        mean = iterativeMean<float64>(i + 1, score, mean);
    }

    return mean;
}

MeasureStoppingCriterion::MeasureStoppingCriterion(std::shared_ptr<IEvaluationMeasure> measurePtr,
                                                   uint32 updateInterval, uint32 stopInterval, uint32 bufferSize)
    : measurePtr_(measurePtr), updateInterval_(updateInterval), stopInterval_(stopInterval),
      buffer_(RingBuffer<float64>(bufferSize)) {

}

bool MeasureStoppingCriterion::shouldContinue(const IPartition& partition, const IStatistics& statistics,
                                              uint32 numRules) {
    bool result = true;

    if (numRules % updateInterval_ == 0) {
        const BiPartition& biPartition = static_cast<const BiPartition&>(partition);
        float64 score = evaluateOnHoldoutSet(biPartition, statistics, *measurePtr_);
        uint32 numElements = buffer_.getNumElements();

        if (numElements > 0 && numRules % stopInterval_ == 0) {
            RingBuffer<float64>::const_iterator iterator = buffer_.cbegin();
            float64 max = iterator[0];

            for (uint32 i = 1; i < numElements; i++) {
                float64 value = iterator[i];

                if (value > max) {
                    max = value;
                }
            }

            // TODO Apply aggregation function
            // TODO Apply decision function
            result = (score + 0.0001) < max;
            std::cout << numRules << ": (" << score << " + 0.0001) < " << max << " = " << result << "\n";
        }

        buffer_.push(score);
    }

    return result;
}
