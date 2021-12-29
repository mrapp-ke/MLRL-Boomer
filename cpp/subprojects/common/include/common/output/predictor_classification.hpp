/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_sparse.hpp"


/**
 * Defines an interface for all classes that allow to predict whether individual labels of given query examples are
 * relevant or irrelevant using an existing rule-based model.
 */
class IClassificationPredictor : public ISparsePredictor<uint8> {

    public:

        virtual ~IClassificationPredictor() { };

};
