/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/statistics/statistics.hpp"

/**
 * Defines an interface for all classes that provide access to gradients and Hessians which serve as the basis for
 * learning a new boosted rule or refining an existing one.
 */
class IBoostingStatistics : public IStatistics {
    public:

        virtual ~IBoostingStatistics() {};
};
