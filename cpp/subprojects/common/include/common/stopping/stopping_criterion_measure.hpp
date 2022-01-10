/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"
#include "common/data/ring_buffer.hpp"


/**
 * Defines an interface for all classes that allow to aggregate the values that are stored in a buffer.
 */
class IAggregationFunction {

    public:

        virtual ~IAggregationFunction() { };

        /**
         * Aggregates the values that are stored in a buffer.
         *
         * @param begin An iterator to the beginning of the buffer
         * @param end   An iterator to the end of the buffer
         * @return      The aggregated value
         */
        virtual float64 aggregate(RingBuffer<float64>::const_iterator begin,
                                  RingBuffer<float64>::const_iterator end) const = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IAggregationFunction`.
 */
class IAggregationFunctionFactory {

    public:

        virtual ~IAggregationFunctionFactory() { };

        /**
         * Creates and returns a new object of type `IAggregationFunction`.
         *
         * @return An unique pointer to an object of type `IAggregationFunction` that has been created
         */
        virtual std::unique_ptr<IAggregationFunction> create() const = 0;

};

/**
 * Allows to create instances of the type `IAggregationFunction` that aggregate the values that are stored in a buffer
 * by finding the minimum value.
 */
class MinAggregationFunctionFactory final : public IAggregationFunctionFactory {

    public:

        std::unique_ptr<IAggregationFunction> create() const override;

};

/**
 * Allows to create instances of the type `IAggregationFunction` that aggregate the values that are stored in a buffer
 * by finding the maximum value.
 */
class MaxAggregationFunctionFactory final : public IAggregationFunctionFactory {

    public:

        std::unique_ptr<IAggregationFunction> create() const override;

};

/**
 * Allows to create instances of the type `IAggregationFunction` that aggregate the values that are stored in a buffer
 * by calculating the arithmetic mean.
 */
class ArithmeticMeanAggregationFunctionFactory final : public IAggregationFunctionFactory {

    public:

        std::unique_ptr<IAggregationFunction> create() const override;

};

/**
 * Allows to configure a stopping criterion that stops the induction of rules as soon as the quality of a model's
 * predictions for the examples in a holdout set do not improve according to a certain measure.
 *
 * This stopping criterion assesses the performance of the current model after every `updateInterval` rules and stores
 * the resulting quality score in a buffer that keeps track of the last `numRecent` scores. If the capacity of this
 * buffer is already reached, the oldest score is passed to a buffer of size `numPast`. Every `stopInterval` rules, it
 * is decided whether the rule induction should be stopped. For this reason, the `numRecent` scores in the first buffer,
 * as well as the `numPast` scores in the second buffer are aggregated according to a certain `aggregationFunction`. If
 * the percentage improvement, which results from comparing the more recent scores from the first buffer to the older
 * scores from the second buffer, is greater than a certain `minImprovement`, the rule induction is continued,
 * otherwise it is stopped.
 */
class MeasureStoppingCriterionConfig final : public IStoppingCriterionConfig {

    public:

        /**
         * Specifies different types of aggregation functions that allow to aggregate the values that are stored in a
         * buffer.
         */
        enum AggregationFunction {

            /**
             * An aggregation function that finds the minimum value in a buffer.
             */
            MIN,

            /**
             * An aggregation function that finds the maximum value in a buffer.
             */
            MAX,

            /**
             * An aggregation function that calculates the arithmetic mean of the values in a buffer.
             */
            ARITHMETIC_MEAN

        };

    private:

        AggregationFunction aggregationFunction_;

        uint32 minRules_;

        uint32 updateInterval_;

        uint32 stopInterval_;

        uint32 numPast_;

        uint32 numCurrent_;

        float64 minImprovement_;

        bool forceStop_;

    public:

        MeasureStoppingCriterionConfig();

        /**
         * Returns the type of the aggregation function that is used to aggregate the values that are stored in a
         * buffer.
         *
         * @return A value of the enum `AggregationFunction` that specifies the type of the aggregation function that is
         *         used to aggregate the values that are stored in a buffer
         */
        AggregationFunction getAggregationFunction() const;

        /**
         * Sets the type of the aggregation function that should be used to aggregate the values that are stored in a
         * buffer.
         *
         * @param aggregationFunction   A value of the enum `AggregationFunction` that specifies the type of the
         *                              aggregation function that should be used to aggregate the values that are stored
         *                              in a buffer
         * @return                      A reference to an object of type `MeasureStoppingCriterionConfig` that allows
         *                              further configuration of the stopping criterion
         */
        MeasureStoppingCriterionConfig& setAggregationFunction(AggregationFunction aggregationFunction);

        /**
         * Returns the minimum number of rules that must have been learned until the induction of rules might be
         * stopped.
         *
         * @return The minimum number of rules that must have been learned until the induction of rules might be stopped
         */
        uint32 getMinRules() const;

        /**
         * Sets the minimum number of rules that must have been learned until the induction of rules might be stopped.
         *
         * @param minRules  The minimum number of rules that must have been learned until the induction of rules might
         *                  be stopped
         * @return          A reference to an object of type `MeasureStoppingCriterionConfig` that allows further
         *                  configuration of the stopping criterion
         */
        MeasureStoppingCriterionConfig& setMinRules(uint32 minRules);

        /**
         * Returns the interval that is used to update the quality of the current model.
         *
         * @return The interval that is used to update the quality of the current model
         */
        uint32 getUpdateInterval() const;

        /**
         * Sets the interval that should be used to update the quality of the current model.
         *
         * @param updateInterval    The interval that should be used to update the quality of the current model, e.g., a
         *                          value of 5 means that the model quality is assessed every 5 rules. Must be at least
         *                          1
         * @return                  A reference to an object of type `MeasureStoppingCriterionConfig` that allows
         *                          further configuration of the stopping criterion
         */
        MeasureStoppingCriterionConfig& setUpdateInterval(uint32 updateInterval);

        /**
         * Returns the interval that is used to decide whether the induction of rules should be stopped.
         *
         * @return The interval that is used to decide whether the induction of rules should be stopped
         */
        uint32 getStopInterval() const;

        /**
         * Sets the interval that should be used to decide whether the induction of rules should be stopped.
         *
         * @param stopInterval  The interval that should be used to decide whether the induction of rules should be
         *                      stopped, e.g., a value of 10 means that the rule induction might be stopped after 10,
         *                      20, ... rules. Must be a multiple of the update interval
         * @return              A reference to an object of type `MeasureStoppingCriterionConfig` that allows further
         *                      configuration of the stopping criterion
         */
        MeasureStoppingCriterionConfig& setStopInterval(uint32 stopInterval);

        /**
         * Returns the number of quality stores of past iterations that are stored in a buffer.
         *
         * @return The number of quality stores of past iterations that are stored in a buffer
         */
        uint32 getNumPast() const;

        /**
         * Sets the number of quality scores of past iterations that should be stored in a buffer.
         *
         * @param numPast   The number of quality scores of past iterations that should be be stored in a buffer. Must
         *                  be at least 1
         * @return          A reference to an object of type `MeasureStoppingCriterionConfig` that allows further
         *                  configuration of the stopping criterion
         */
        MeasureStoppingCriterionConfig& setNumPast(uint32 numPast);

        /**
         * Returns the number of quality scores of the most recent iterations that are stored in a buffer.
         *
         * @return The number of quality scores of the most recent iterations that are stored in a buffer
         */
        uint32 getNumCurrent() const;

        /**
         * Sets the number of quality scores of the most recent iterations that should be stored in a buffer.
         *
         * @param numCurrent    The number of quality scores of the most recent iterations that should be stored in a
         *                      buffer. Must be at least 1
         * @return              A reference to an object of type `MeasureStoppingCriterionConfig` that allows further
         *                      configuration of the stopping criterion
         */
        MeasureStoppingCriterionConfig& setNumCurrent(uint32 numCurrent);

        /**
         * Returns the minimum improvement that must be reached for the rule induction to be continued.
         *
         * @return The minimum improvement that must be reached for the rule induction to be continued
         */
        float64 getMinImprovement() const;

        /**
         * Sets the minimum improvement that must be reached for the rule induction to be continued.
         *
         * @param minImprovement    The minimum improvement in percent that must be reached for the rule induction to be
         *                          continued. Must be in [0, 1]
         * @return                  A reference to an object of type `MeasureStoppingCriterionConfig` that allows
         *                          further configuration of the stopping criterion
         */
        MeasureStoppingCriterionConfig& setMinImprovement(float64 minImprovement);

        /**
         * Returns whether the induction of rules is forced to be stopped, if the stopping criterion is met.
         *
         * @return True, if the induction of rules is forced to be stopped, if the stopping criterion is met, false, if
         *         only the time of stopping is stored
         */
        bool getForceStop() const;

        /**
         * Sets whether the induction of rules should be forced to be stopped, if the stopping criterion is met.
         *
         * @param forceStop True, if the induction of rules should be forced to be stopped, if the stopping criterion is
         *                  met, false, if only the time of stopping should be stored
         * @return          A reference to an object of type `MeasureStoppingCriterionConfig` that allows further
         *                  configuration of the stopping criterion
         */
        MeasureStoppingCriterionConfig& setForceStop(bool forceStop);

};

/**
 * Allows to create implementations of the type `IStoppingCriterion` that stop the induction of rules as soon as the
 * quality of a model's predictions for the examples in a holdout set do not improve according a certain measure.
 */
class MeasureStoppingCriterionFactory final : public IStoppingCriterionFactory {

    private:

        std::unique_ptr<IAggregationFunctionFactory> aggregationFunctionFactoryPtr_;

        uint32 minRules_;

        uint32 updateInterval_;

        uint32 stopInterval_;

        uint32 numPast_;

        uint32 numCurrent_;

        float64 minImprovement_;

        bool forceStop_;

    public:

        /**
         * @param aggregationFunctionFactoryPtr An unique pointer to an object of type `IAggregationFunctionFactory`
         *                                      that allows to create implementations of the aggregation function that
         *                                      should be used to aggregate the scores in the buffer
         * @param minRules                      The minimum number of rules that must have been learned until the
         *                                      induction of rules might be stopped. Must be at least 1
         * @param updateInterval                The interval to be used to update the quality of the current model,
         *                                      e.g., a value of 5 means that the model quality is assessed every 5
         *                                      rules. Must be at least 1
         * @param stopInterval                  The interval to be used to decide whether the induction of rules should
         *                                      be stopped, e.g., a value of 10 means that the rule induction might be
         *                                      stopped after 10, 20, ... rules. Must be a multiple of `updateInterval`
         * @param numPast                       The number of quality scores of past iterations to be stored in a
         *                                      buffer. Must be at least 1
         * @param numCurrent                    The number of quality scores of the most recent iterations to be stored
         *                                      in a buffer. Must be at least 1
         * @param minImprovement                The minimum improvement in percent that must be reached for the rule
         *                                      induction to be continued. Must be in [0, 1]
         * @param forceStop                     True, if the induction of rules should be forced to be stopped, if the
         *                                      stopping criterion is met, false, if only the time of stopping should be
         *                                      stored
         */
        MeasureStoppingCriterionFactory(std::unique_ptr<IAggregationFunctionFactory> aggregationFunctionFactoryPtr,
                                        uint32 minRules, uint32 updateInterval, uint32 stopInterval, uint32 numPast,
                                        uint32 numCurrent, float64 minImprovement, bool forceStop);

        std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override;

        std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override;

};

/**
 * Creates and returns a new object of type `IAggregationFunctionFactory`.
 *
 * @param aggregationFunction   A value of the enum `MeasureStoppingCriterionConfig::AggregationFunction` that specifies
 *                              the type of the aggregation function
 * @return                      An unique pointer to an object of type `IAggregationFunctionFactory` that has been
 *                              created
 */
std::unique_ptr<IAggregationFunctionFactory> createAggregationFunctionFactory(
    MeasureStoppingCriterionConfig::AggregationFunction aggregationFunction);
