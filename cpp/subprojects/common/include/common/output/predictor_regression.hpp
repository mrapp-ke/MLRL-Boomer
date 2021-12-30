/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor.hpp"
#include "common/model/rule_list.hpp"


/**
 * Defines an interface for all classes that allow predict label-wise regression scores for given query examples using
 * an existing rule-based model.
 */
class IRegressionPredictor : public IPredictor<float64> {

    public:

        virtual ~IRegressionPredictor() { };

        /**
         * Obtains predictions for all examples in a C-contiguous matrix, using a single rule, and writes them to a
         * given prediction matrix.
         *
         * @param featureMatrix     A reference to an object of type `CContiguousFeatureMatrix` that stores the feature
         *                          values of the examples
         * @param predictionMatrix  A reference to an object of type `CContiguousView`, the predictions should be
         *                          written to. May contain arbitrary values
         * @param rule              A reference to an object of type `Rule` that should be used to obtain the
         *                          predictions
         * @param labelVectors      A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         */
        virtual void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                             const Rule& rule, const LabelVectorSet* labelVectors) const = 0;

        /**
         * Obtains predictions for all examples in a sparse CSR matrix, using a single rule, and writes them to a given
         * prediction matrix.
         *
         * @param featureMatrix     A reference to an object of type `CsrFeatureMatrix` that stores the feature values
         *                          of the examples
         * @param predictionMatrix  A reference to an object of type `CContiguousView`, the predictions should be
         *                          written to. May contain arbitrary values
         * @param rule              A reference to an object of type `Rule` that should be used to obtain the
         *                          predictions
         * @param labelVectors      A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         */
        virtual void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                             const Rule& rule, const LabelVectorSet* labelVectors) const = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IRegressionPredictor`.
 */
class IRegressionPredictorFactory {

    public:

        virtual ~IRegressionPredictorFactory() { };

        /**
         * Creates and returns a new object of the type `IRegressionPredictor`.
         *
         * @param model A reference to an object of type `RuleList` that should be used to obtain the predictions
         * @return      An unique pointer to an object of type `IRegressionPredictor` that has been created
         */
        virtual std::unique_ptr<IRegressionPredictor> create(const RuleList& model) const = 0;

};
