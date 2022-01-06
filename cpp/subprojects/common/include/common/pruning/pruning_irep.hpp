/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/pruning/pruning.hpp"


/**
 * Allows to create instances of the type `IPruning` that prune rules by following the ideas of "incremental reduced
 * error pruning" (IREP). Given `n` conditions in the order of their induction, IREP may remove up to `n - 1` trailing
 * conditions, depending on which of the resulting rules comes with the greatest improvement in terms of quality as
 * measured on the prune set.
 */
class IrepFactory final : virtual public IPruningFactory {

    public:

        std::unique_ptr<IPruning> create() const override;

};
