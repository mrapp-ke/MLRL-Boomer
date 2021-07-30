/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/pruning/pruning.hpp"
#include "seco/head_refinement/lift_function.hpp"
#include "common/data/vector_sparse_array.hpp"
#include "common/rule_evaluation/score_processor_label_wise.hpp"


/**
 * Implements incremental reduced error pruning (IREP) for pruning classification rules.
 *
 * Given `n` conditions in the order of their induction, IREP allows to remove up to `n - 1` trailing conditions,
 * depending on which of the resulting rules improves the most over the quality score of the original rules as measured
 * on the prune set.
 */

namespace seco {

    class SecoPruning final : public IPruning, public ILabelWiseScoreProcessor {

    private:

        std::shared_ptr<seco::ILiftFunction> liftFunctionPtr_;

        std::unique_ptr<PartialPrediction> headPtr_;

        template<typename ScoreVector>
        const AbstractEvaluatedPrediction *
        findHeadInternally(const AbstractEvaluatedPrediction *bestHead, const ScoreVector &scoreVector);

    public:

        explicit SecoPruning(std::shared_ptr<seco::ILiftFunction> liftFunctionPtr);

        std::unique_ptr<ICoverageState> prune(IThresholdsSubset &thresholdsSubset, IPartition &partition,
                                              ConditionList &conditions, AbstractEvaluatedPrediction* bestHead) const override;

        const AbstractEvaluatedPrediction *processScores(const AbstractEvaluatedPrediction *bestHead,
                                                         const DenseLabelWiseScoreVector<FullIndexVector> &scoreVector) override;

        const AbstractEvaluatedPrediction *processScores(const AbstractEvaluatedPrediction *bestHead,
                                                         const DenseLabelWiseScoreVector<PartialIndexVector> &scoreVector) override;

        const AbstractEvaluatedPrediction *processScores(const AbstractEvaluatedPrediction *bestHead,
                                                         const DenseBinnedLabelWiseScoreVector<FullIndexVector> &scoreVector) override;

        const AbstractEvaluatedPrediction *processScores(const AbstractEvaluatedPrediction *bestHead,
                                                         const DenseBinnedLabelWiseScoreVector<PartialIndexVector> &scoreVector) override;
    };
}