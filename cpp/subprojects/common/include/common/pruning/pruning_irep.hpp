/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/pruning/pruning.hpp"


/**
 * Defines an interface for all classes that allow to configure a strategy for pruning classification rules that prunes
 * rules by following the ideas of "incremental reduced error pruning" (IREP).
 */
class IIrepConfig {

    public:

        virtual ~IIrepConfig() { };

};

/**
 * Allows to configure a strategy for pruning classification rules that prunes rules by following the ideas of
 * "incremental reduced error pruning" (IREP).
 */
class IrepConfig final : public IPruningConfig, public IIrepConfig {

    public:

        std::unique_ptr<IPruningFactory> configure() const override;

};

