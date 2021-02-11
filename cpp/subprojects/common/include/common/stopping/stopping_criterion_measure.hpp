/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"
#include "common/measures/measure_evaluation.hpp"
#include "common/data/ring_buffer.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to aggregate the values that are stored in a buffer.
 */
class IAggregationFunction {

    public:

        virtual ~IAggregationFunction() { };

        /**
         * Aggregates the values that are stored in a buffer.
         *
         * @param numElements   The number of values that are stored in the buffer
         * @param iterator      An iterator that provides access to the values that are stored in the buffer
         * @return              The aggregated value
         */
        virtual float64 aggregate(uint32 numElements, RingBuffer<float64>::const_iterator iterator) const = 0;

};

/**
 * Allows to aggregate the values that are stored in a buffer by finding the minimum value.
 */
class MinFunction : public IAggregationFunction {

    public:

        float64 aggregate(uint32 numElements, RingBuffer<float64>::const_iterator iterator) const override;

};

/**
 * Allows to aggregate the values that are stored in a buffer by finding the maximum value.
 */
class MaxFunction : public IAggregationFunction {

    public:

        float64 aggregate(uint32 numElements, RingBuffer<float64>::const_iterator iterator) const override;

};

/**
 * Allows to aggregate the values that are stored in a buffer by calculating the arithmetic mean.
 */
class ArithmeticMeanFunction : public IAggregationFunction {

    public:

        float64 aggregate(uint32 numElements, RingBuffer<float64>::const_iterator iterator) const override;

};

/**
 * A stopping criterion that stops the induction of rules as soon as the quality of a model's predictions for the
 * examples in a holdout set do not improve according a certain measure.
 *
 * This stopping criterion assesses the performance of the current model after every `updateInterval` rules and stores
 * the resulting quality score in a buffer that keeps track of the last `bufferSize` scores. Every `stopInterval` rules,
 * it is decided whether the rule induction should be stopped. For this reason, the scores in the buffer are aggregated
 * according to an `aggregationFunction`. If the percentage improvement between the aggregated score and the current
 * score is greater than a certain `minImprovement`, the rule induction is continued, otherwise it is stopped.
 */
class MeasureStoppingCriterion final : public IStoppingCriterion {

    private:

        std::shared_ptr<IEvaluationMeasure> measurePtr_;

        std::shared_ptr<IAggregationFunction> aggregationFunctionPtr_;

        uint32 minRules_;

        uint32 updateInterval_;

        uint32 stopInterval_;

        float64 minImprovement_;

        RingBuffer<float64> buffer_;

        uint32 offset_;

        Result stoppingResult_;

    public:

        /**
         * @param measurePtr                A shared pointer to an object of type `IEvaluationMeasure` that should be
         *                                  used to assess the quality of a model
         * @param aggregationFunctionPtr    A shared pointer to an object of type `IAggregationFunction` that should be
         *                                  used to aggregate the scores in the buffer
         * @param minRules                  The minimum number of rules that must have been learned until the induction
         *                                  of rules might be stopped. Must be at least 1
         * @param updateInterval            The interval to be used to update the quality of the current model, e.g., a
         *                                  value of 5 means that the model quality is assessed every 5 rules. Must be
         *                                  at least 1
         * @param stopInterval              The interval to be used to decide whether the induction of rules should be
         *                                  stopped, e.g., a value of 10 means that the rule induction might be stopped
         *                                  after 10, 20, ... rules. Must be a multiple of `updateInterval`
         * @param bufferSize                The number of quality scores to be stored in a buffer. Must be at least 1
         * @param minImprovement            The minimum improvement in percent that must be reached for the rule
         *                                  induction to be continued. Must be in [0, 1]
         * @param forceStop                 True, if the induction of rules should be forced to be stopped, if the
         *                                  stopping criterion is met, false, if the time of stopping should only be
         *                                  stored
         */
        MeasureStoppingCriterion(std::shared_ptr<IEvaluationMeasure> measurePtr,
                                 std::shared_ptr<IAggregationFunction> aggregationFunctionPtr, uint32 minRules,
                                 uint32 updateInterval, uint32 stopInterval, uint32 bufferSize, float64 minImprovement,
                                 bool forceStop);

        Result test(const IPartition& partition, const IStatistics& statistics, uint32 numRules) override;

};
