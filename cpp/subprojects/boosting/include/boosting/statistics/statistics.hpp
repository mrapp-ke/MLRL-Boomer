/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/statistics/statistics.hpp"

/**
 * Defines an interface for all classes that store gradients and Hessians.
 */
class IBoostingStatistics : public IStatistics {
    public:

        virtual ~IBoostingStatistics() {};
};
