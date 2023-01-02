/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"


/**
 * Defines an interface for all classes that allow to configure a stopping criterion that allows to decide how many
 * rules should be included in a model, such that its performance is optimized globally.
 */
class IGlobalPruningConfig : public IStoppingCriterionConfig {

    public:

        virtual ~IGlobalPruningConfig() override { };

        virtual bool shouldRemoveUnusedRules() const = 0;

};
