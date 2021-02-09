/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"
#include "common/measures/measure.hpp"
#include <memory>


/**
 * A stopping criterion that stops the induction of rules as soon as the quality of a model's predictions for the
 * examples in a validation set do not improve according a certain measure.
 */
class MeasureStoppingCriterion final : public IStoppingCriterion {

    private:

        std::shared_ptr<IMeasure> measurePtr_;

    public:

        /**
         * @param measurePtr A shared pointer to an object of type `IMeasure` that should be used to assess the quality
         *                   of predictions
         */
        MeasureStoppingCriterion(std::shared_ptr<IMeasure> measurePtr);

        bool shouldContinue(const IStatistics& statistics, uint32 numRules) override;

};
