/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/pruning/pruning.hpp"


/**
 * Allows to create instances of the type `IPruning` that do not actually perform any pruning.
 */
class NoPruningFactory final : virtual public IPruningFactory {

    public:

        std::unique_ptr<IPruning> create() const override;

};
