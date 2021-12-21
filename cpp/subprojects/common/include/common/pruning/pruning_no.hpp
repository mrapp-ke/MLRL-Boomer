/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/pruning/pruning.hpp"


/**
 * An implementation of the class `IPruning` that does not actually perform any pruning.
 */
class NoPruning final : public IPruning {

    public:

        std::unique_ptr<ICoverageState> prune(IThresholdsSubset& thresholdsSubset, IPartition& partition,
                                              ConditionList& conditions, const AbstractPrediction& head) const override;

};

/**
 * Allows to create instances of the type `IPruning` that do not actually perform any pruning.
 */
class NoPruningFactory final : public IPruningFactory {

    public:

        std::unique_ptr<IPruning> create() const override;

};
