/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csr.hpp"
#include "mlrl/common/model/rule_list.hpp"
#include "mlrl/common/prediction/label_vector_set.hpp"
#include "mlrl/common/prediction/prediction_matrix_dense.hpp"
#include "mlrl/common/prediction/predictor.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow to predict scores for given query examples.
 */
class IScorePredictor : public IPredictor<DensePredictionMatrix<float64>> {
    public:

        virtual ~IScorePredictor() override {}
};

/**
 * Defines an interface for all classes that allow to create instances of the type `IScorePredictor`.
 */
class IScorePredictorFactory {
    public:

        virtual ~IScorePredictorFactory() {}

        /**
         * Creates and returns a new object of the type `IScorePredictor`.
         *
         * @param featureMatrix     A reference to an object of type `CContiguousView` that stores the feature values of
         *                          the query examples to predict for
         * @param model             A reference to an object of type `RuleList` that should be used to obtain
         *                          predictions
         * @param labelVectorSet    A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @param numOutputs        The number of outputs to predict for
         * @return                  An unique pointer to an object of type `IScorePredictor` that has been created
         */
        virtual std::unique_ptr<IScorePredictor> create(const CContiguousView<const float32>& featureMatrix,
                                                        const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                        uint32 numOutputs) const = 0;

        /**
         * Creates and returns a new object of the type `IScorePredictor`.
         *
         * @param featureMatrix     A reference to an object of type `CsrView` that stores the feature values of the
         *                          query examples to predict for
         * @param model             A reference to an object of type `RuleList` that should be used to obtain
         *                          predictions
         * @param labelVectorSet    A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @param numOutputs        The number of outputs to predict for
         * @return                  An unique pointer to an object of type `IScorePredictor` that has been created
         */
        virtual std::unique_ptr<IScorePredictor> create(const CsrView<const float32>& featureMatrix,
                                                        const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                        uint32 numOutputs) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a `IScorePredictor`.
 */
class IScorePredictorConfig : public IPredictorConfig<IScorePredictorFactory> {
    public:

        virtual ~IScorePredictorConfig() override {}
};
