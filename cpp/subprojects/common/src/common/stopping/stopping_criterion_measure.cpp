#include "common/stopping/stopping_criterion_measure.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/math/math.hpp"


static inline float64 evaluateOnHoldoutSet(const BiPartition& partition, const IStatistics& statistics,
                                           const IMeasure& measure) {
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

MeasureStoppingCriterion::MeasureStoppingCriterion(std::shared_ptr<IMeasure> measurePtr, uint32 updateInterval,
                                                   uint32 stopInterval)
    : measurePtr_(measurePtr), updateInterval_(updateInterval), stopInterval_(stopInterval) {

}

bool MeasureStoppingCriterion::shouldContinue(const IPartition& partition, const IStatistics& statistics,
                                              uint32 numRules) {
    const BiPartition& biPartition = static_cast<const BiPartition&>(partition);
    float64 score = evaluateOnHoldoutSet(biPartition, statistics, *measurePtr_);
    // TODO Implement
    return true;
}
