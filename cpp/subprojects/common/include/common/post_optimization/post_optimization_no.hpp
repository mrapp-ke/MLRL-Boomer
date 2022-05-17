/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/post_optimization/post_optimization.hpp"


/**
 * Allows to configure a method that does not perform any optimizations, but retains a previously learned rule-based
 * model.
 */
class NoPostOptimizationConfig final : public IPostOptimizationConfig {

    public:

        std::unique_ptr<IPostOptimizationFactory> createPostOptimizationFactory() const override;

};
