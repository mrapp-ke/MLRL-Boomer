#include "common/stopping/stopping_criterion_measure.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/math/math.hpp"
#include "common/util/validation.hpp"
#include <limits>


/**
 * An implementation of the type `IAggregationFunction` that aggregates the values that are stored in a buffer by
 * finding the minimum value.
 */
class MinAggregationFunction final : public IAggregationFunction {

    public:

        float64 aggregate(RingBuffer<float64>::const_iterator begin,
                          RingBuffer<float64>::const_iterator end) const override {
            uint32 numElements = end - begin;
            float64 min = begin[0];

            for (uint32 i = 1; i < numElements; i++) {
                float64 value = begin[i];

                if (value < min) {
                    min = value;
                }
            }

            return min;
        }

};

/**
 * An implementation of the type `IAggregationFunction` that aggregates the values that are stored in a buffer by
 * finding the maximum value.
 */
class MaxAggregationFunction final : public IAggregationFunction {

    public:

        float64 aggregate(RingBuffer<float64>::const_iterator begin,
                          RingBuffer<float64>::const_iterator end) const override {
            uint32 numElements = end - begin;
            float64 max = begin[0];

            for (uint32 i = 1; i < numElements; i++) {
                float64 value = begin[i];

                if (value > max) {
                    max = value;
                }
            }

            return max;
        }

};

/**
 * An implementation of the type `IAggregationFunction` that aggregates the values that are stored in a buffer by
 * calculating the arithmetic mean.
 */
class ArithmeticMeanAggregationFunction final : public IAggregationFunction {

    public:

        float64 aggregate(RingBuffer<float64>::const_iterator begin,
                          RingBuffer<float64>::const_iterator end) const override {
            uint32 numElements = end - begin;
            float64 mean = 0;

            for (uint32 i = 0; i < numElements; i++) {
                float64 value = begin[i];
                mean = iterativeArithmeticMean<float64>(i + 1, value, mean);
            }

            return mean;
        }

};

static inline float64 evaluate(const SinglePartition& partition, const IStatistics& statistics) {
    uint32 numExamples = partition.getNumElements();
    SinglePartition::const_iterator iterator = partition.cbegin();
    float64 mean = 0;

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = iterator[i];
        float64 score = statistics.evaluatePrediction(exampleIndex);
        mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
    }

    return mean;
}

static inline float64 evaluate(const BiPartition& partition, const IStatistics& statistics) {
    uint32 numHoldoutExamples = partition.getNumSecond();
    BiPartition::const_iterator iterator = partition.second_cbegin();
    float64 mean = 0;

    for (uint32 i = 0; i < numHoldoutExamples; i++) {
        uint32 exampleIndex = iterator[i];
        float64 score = statistics.evaluatePrediction(exampleIndex);
        mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
    }

    return mean;
}

/**
 * An implementation of the type `IStoppingCriterion` that stops the induction of rules as soon as the quality of a
 * model's predictions for the examples in a holdout set do not improve according a certain measure.
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   training set
 */
template<typename Partition>
class MeasureStoppingCriterion final : public IStoppingCriterion {

    private:

        std::unique_ptr<IAggregationFunction> aggregationFunctionPtr_;

        Partition partition_;

        uint32 updateInterval_;

        uint32 stopInterval_;

        float64 minImprovement_;

        RingBuffer<float64> pastBuffer_;

        RingBuffer<float64> recentBuffer_;

        uint32 offset_;

        Action stoppingAction_;

        float64 bestScore_;

        uint32 bestNumRules_;

        bool stopped_;

    public:

        /**
         * @param aggregationFunctionPtr    An unique pointer to an object of type `IAggregationFunctionFactory` that
         *                                  allows to create implementations of the aggregation function that should be
         *                                  used to aggregate the scores in the buffer
         * @param minRules                  The minimum number of rules that must have been learned until the induction
         *                                  of rules might be stopped. Must be at least 1
         * @param updateInterval            The interval to be used to update the quality of the current model, e.g., a
         *                                  value of 5 means that the model quality is assessed every 5 rules. Must be
         *                                  at least 1
         * @param stopInterval              The interval to be used to decide whether the induction of rules should be
         *                                  stopped, e.g., a value of 10 means that the rule induction might be stopped
         *                                  after 10, 20, ... rules. Must be a multiple of `updateInterval`
         * @param numPast                   The number of quality scores of past iterations to be stored in a buffer.
         *                                  Must be at least 1
         * @param numCurrent                The number of quality scores of the most recent iterations to be stored in a
         *                                  buffer. Must be at least 1
         * @param minImprovement            The minimum improvement in percent that must be reached for the rule
         *                                  induction to be continued. Must be in [0, 1]
         * @param forceStop                 True, if the induction of rules should be forced to be stopped, if the
         *                                  stopping criterion is met, false, if the time of stopping should only be
         *                                  stored
         */
        MeasureStoppingCriterion(std::unique_ptr<IAggregationFunction> aggregationFunctionPtr, Partition& partition,
                                 uint32 minRules, uint32 updateInterval, uint32 stopInterval, uint32 numPast,
                                 uint32 numCurrent, float64 minImprovement, bool forceStop)
            : aggregationFunctionPtr_(std::move(aggregationFunctionPtr)), partition_(partition),
              updateInterval_(updateInterval), stopInterval_(stopInterval), minImprovement_(minImprovement),
              pastBuffer_(RingBuffer<float64>(numPast)), recentBuffer_(RingBuffer<float64>(numCurrent)),
              stoppingAction_(forceStop ? FORCE_STOP : STORE_STOP),
              bestScore_(std::numeric_limits<float64>::infinity()), stopped_(false) {
            uint32 bufferInterval = (numPast * updateInterval) + (numCurrent * updateInterval);
            offset_ = bufferInterval < minRules ? minRules - bufferInterval : 0;
        }

        Result test(const IStatistics& statistics, uint32 numRules) override {
            Result result;
            result.action = CONTINUE;

            if (!stopped_ && numRules > offset_ && numRules % updateInterval_ == 0) {
                float64 currentScore = evaluate(partition_, statistics);

                if (pastBuffer_.isFull()) {
                    if (currentScore < bestScore_) {
                        bestScore_ = currentScore;
                        bestNumRules_ = numRules;
                    }

                    if (numRules % stopInterval_ == 0) {
                        float64 aggregatedScorePast =
                            aggregationFunctionPtr_->aggregate(pastBuffer_.cbegin(), pastBuffer_.cend());
                        float64 aggregatedScoreRecent =
                            aggregationFunctionPtr_->aggregate(recentBuffer_.cbegin(), recentBuffer_.cend());
                        float64 percentageImprovement =
                            (aggregatedScorePast - aggregatedScoreRecent) / aggregatedScoreRecent;

                        if (percentageImprovement <= minImprovement_) {
                            result.action = stoppingAction_;
                            result.numRules = bestNumRules_;
                            stopped_ = true;
                        }
                    }
                }

                std::pair<bool, float64> pair = recentBuffer_.push(currentScore);

                if (pair.first) {
                    pastBuffer_.push(pair.second);
                }
            }

            return result;
        }

};

MeasureStoppingCriterionConfig::MeasureStoppingCriterionConfig()
    : aggregationFunction_(MeasureStoppingCriterionConfig::AggregationFunction::ARITHMETIC_MEAN), minRules_(100),
      updateInterval_(1), stopInterval_(1), numPast_(50), numCurrent_(50), minImprovement_(0.005), forceStop_(true) {

}

MeasureStoppingCriterionConfig::AggregationFunction MeasureStoppingCriterionConfig::getAggregationFunction() const {
    return aggregationFunction_;
}

MeasureStoppingCriterionConfig& MeasureStoppingCriterionConfig::setAggregationFunction(
        MeasureStoppingCriterionConfig::AggregationFunction aggregationFunction) {
    aggregationFunction_ = aggregationFunction;
    return *this;
}

uint32 MeasureStoppingCriterionConfig::getMinRules() const {
    return minRules_;
}

MeasureStoppingCriterionConfig& MeasureStoppingCriterionConfig::setMinRules(uint32 minRules) {
    assertGreaterOrEqual<uint32>("minRules", minRules, 1);
    minRules_ = minRules;
    return *this;
}

uint32 MeasureStoppingCriterionConfig::getUpdateInterval() const {
    return updateInterval_;
}

MeasureStoppingCriterionConfig& MeasureStoppingCriterionConfig::setUpdateInterval(uint32 updateInterval) {
    assertGreaterOrEqual<uint32>("updateInterval", updateInterval, 1);
    updateInterval_ = updateInterval;
    return *this;
}

uint32 MeasureStoppingCriterionConfig::getStopInterval() const {
    return stopInterval_;
}

MeasureStoppingCriterionConfig& MeasureStoppingCriterionConfig::setStopInterval(uint32 stopInterval) {
    assertMultiple<uint32>("stopInterval", stopInterval, updateInterval_);
    stopInterval_ = stopInterval;
    return *this;
}

uint32 MeasureStoppingCriterionConfig::getNumPast() const {
    return numPast_;
}

MeasureStoppingCriterionConfig& MeasureStoppingCriterionConfig::setNumPast(uint32 numPast) {
    assertGreaterOrEqual<uint32>("numPast", numPast, 1);
    numPast_ = numPast;
    return *this;
}

uint32 MeasureStoppingCriterionConfig::getNumCurrent() const {
    return numCurrent_;
}

MeasureStoppingCriterionConfig& MeasureStoppingCriterionConfig::setNumCurrent(uint32 numCurrent) {
    assertGreaterOrEqual<uint32>("numCurrent", numCurrent, 1);
    numCurrent_ = numCurrent;
    return *this;
}

float64 MeasureStoppingCriterionConfig::getMinImprovement() const {
    return minImprovement_;
}

MeasureStoppingCriterionConfig& MeasureStoppingCriterionConfig::setMinImprovement(float64 minImprovement) {
    assertGreaterOrEqual<float64>("minImprovement", minImprovement, 0);
    assertLessOrEqual<float64>("minImprovement", minImprovement, 1);
    minImprovement_ = minImprovement;
    return *this;
}

bool MeasureStoppingCriterionConfig::getForceStop() const{
    return forceStop_;
}

MeasureStoppingCriterionConfig& MeasureStoppingCriterionConfig::setForceStop(bool forceStop) {
    forceStop_ = forceStop;
    return *this;
}

std::unique_ptr<IAggregationFunction> MinAggregationFunctionFactory::create() const {
    return std::make_unique<MinAggregationFunction>();
}

std::unique_ptr<IAggregationFunction> MaxAggregationFunctionFactory::create() const {
    return std::make_unique<MaxAggregationFunction>();
}

std::unique_ptr<IAggregationFunction> ArithmeticMeanAggregationFunctionFactory::create() const {
    return std::make_unique<ArithmeticMeanAggregationFunction>();
}

MeasureStoppingCriterionFactory::MeasureStoppingCriterionFactory(
        std::unique_ptr<IAggregationFunctionFactory> aggregationFunctionFactoryPtr, uint32 minRules,
        uint32 updateInterval, uint32 stopInterval, uint32 numPast, uint32 numCurrent, float64 minImprovement,
        bool forceStop)
    : aggregationFunctionFactoryPtr_(std::move(aggregationFunctionFactoryPtr)), minRules_(minRules),
      updateInterval_(updateInterval), stopInterval_(stopInterval), numPast_(numPast), numCurrent_(numCurrent),
      minImprovement_(minImprovement), forceStop_(forceStop) {

}

std::unique_ptr<IStoppingCriterion> MeasureStoppingCriterionFactory::create(const SinglePartition& partition) const {
    std::unique_ptr<IAggregationFunction> aggregationFunctionPtr = aggregationFunctionFactoryPtr_->create();
    return std::make_unique<MeasureStoppingCriterion<const SinglePartition>>(std::move(aggregationFunctionPtr),
                                                                             partition, minRules_, updateInterval_,
                                                                             stopInterval_, numPast_, numCurrent_,
                                                                             minImprovement_, forceStop_);
}

std::unique_ptr<IStoppingCriterion> MeasureStoppingCriterionFactory::create(BiPartition& partition) const {
    std::unique_ptr<IAggregationFunction> aggregationFunctionPtr = aggregationFunctionFactoryPtr_->create();
    return std::make_unique<MeasureStoppingCriterion<BiPartition>>(std::move(aggregationFunctionPtr), partition,
                                                                   minRules_, updateInterval_, stopInterval_, numPast_,
                                                                   numCurrent_, minImprovement_, forceStop_);
}
