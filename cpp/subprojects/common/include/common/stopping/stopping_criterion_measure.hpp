/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"
#include "common/measures/measure_evaluation.hpp"
#include "common/data/ring_buffer.hpp"
#include <memory>


/**
 * A stopping criterion that stops the induction of rules as soon as the quality of a model's predictions for the
 * examples in a holdout set do not improve according a certain measure.
 *
 * This stopping criterion assesses the performance of the current model after every `updateInterval` rules and stores
 * the resulting quality score in a buffer that keeps track of the last `bufferSize` scores. Every `stopInterval` rules,
 * it is decided whether the rule induction should be stopped by aggregating the scores in the buffer according to a
 * `aggregationFunction` and comparing it to the latest quality score using a `decisionFunction`.
 */
class MeasureStoppingCriterion final : public IStoppingCriterion {

    private:

        std::shared_ptr<IEvaluationMeasure> measurePtr_;

        uint32 minRules_;

        uint32 updateInterval_;

        uint32 stopInterval_;

        RingBuffer<float64> buffer_;

        uint32 offset_;

    public:

        /**
         * @param measurePtr        A shared pointer to an object of type `IEvaluationMeasure` that should be used to
         *                          assess the quality of a model
         * @param minRules          The minimum number of rules that must have been learned until the induction of rules
         *                          might be stopped. Must be at least 1
         * @param updateInterval    The interval to be used to update the quality of the current model, e.g., a value of
         *                          5 means that the model quality is assessed every 5 rules
         * @param stopInterval      The interval to be used to decide whether the induction of rules should be stopped,
         *                          e.g., a value of 10 means that the rule induction might be stopped after 10, 20, ...
         *                          rules. Must be a multiple of `updateInterval`
         * @param bufferSize        The number of quality scores to be stored in a buffer. Must be at least 1
         */
        MeasureStoppingCriterion(std::shared_ptr<IEvaluationMeasure> measurePtr, uint32 minRules, uint32 updateInterval,
                                 uint32 stopInterval, uint32 bufferSize);

        bool shouldContinue(const IPartition& partition, const IStatistics& statistics, uint32 numRules) override;

};
