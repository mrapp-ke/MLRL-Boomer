/**
 * Provides classes that implement strategies for pruning classification rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "thresholds.h"
#include <list>


/**
 * Defines an interface for all classes that implement a strategy for pruning classification rules based on a
 * "prune set", i.e., based on the examples that are not contained in the sub-sample of the training data that has been
 * used to learn the rule, referred to a the "grow set".
 */
class IPruning {

    public:

        virtual ~IPruning() { };

        /**
         * Prunes the conditions of an existing rule by modifying a given list of conditions in-place. The rule is
         * pruned by removing individual conditions in a way that improves over its original quality score as measured
         * on the the prune set.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset`, which contains the thresholds
         *                          that correspond to the subspace of the instance space that is covered by the
         *                          existing rule
         * @param conditions        A reference to a list that contains the conditions of the existing rule
         * @param head              A reference to an object of type `AbstractPrediction` that stores the scores that
         *                          are predicted by the existing rule
         * @return                  An unique pointer to an object of type `CoverageMask` that specifies the examples
         *                          that are covered by the pruned rule or a null pointer if the rule was not pruned
         */
        virtual std::unique_ptr<CoverageMask> prune(IThresholdsSubset& thresholdsSubset,
                                                    std::list<Condition>& conditions,
                                                    const AbstractPrediction& head) const = 0;

};

/**
 * Implements incremental reduced error pruning (IREP) for pruning classification rules.
 *
 * Given `n` conditions in the order of their induction, IREP allows to remove up to `n - 1` trailing conditions,
 * depending on which of the resulting rules improves the most over the quality score of the original rules as measured
 * on the prune set.
 */
class IREPImpl : virtual public IPruning {

    public:

        virtual std::unique_ptr<CoverageMask> prune(IThresholdsSubset& thresholdsSubset,
                                                    std::list<Condition>& conditions,
                                                    const AbstractPrediction& head) const override;

};
