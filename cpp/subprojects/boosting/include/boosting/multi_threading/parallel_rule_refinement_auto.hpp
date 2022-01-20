/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/multi_threading/multi_threading.hpp"


/**
 * Allows to configure the multi-threading behavior that is used for the parallel refinement of rules by automatically
 * deciding for the number of threads to be used.
 */
class AutoParallelRuleRefinementConfig final : public IMultiThreadingConfig {

    public:

        uint32 configure() const override;

};
