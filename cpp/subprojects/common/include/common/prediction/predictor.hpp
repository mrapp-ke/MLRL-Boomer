/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/data/view_csr.hpp"
#include "common/input/feature_matrix_row_wise.hpp"
#include "common/model/rule_list.hpp"
#include "common/prediction/label_vector_set.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to make prediction for given query examples.
 *
 * @tparam PredictionMatrix The type of the matrix that is used to store the predictions
 */
template<typename PredictionMatrix>
class IPredictor {

    public:

        virtual ~IPredictor() { };

        /**
         * Obtains and returns predictions for all query examples in a C-contiguous matrix.
         *
         * @param featureMatrix A reference to an object of type `CContiguousConstView` that stores the feature values
         *                      of the query examples
         * @return              An unique pointer to an object of template type `PredictionMatrix` that stores the
         *                      predictions
         */
        virtual std::unique_ptr<PredictionMatrix> predict(
            const CContiguousConstView<const float32>& featureMatrix) const = 0;

        /**
         * Obtains and returns predictions for all query examples in a sparse CSR matrix.
         *
         * @param featureMatrix A reference to an object of type `CsrConstView` that stores the feature values of the
         *                      query examples
         * @return              An unique pointer to an object of template type `PredictionMatrix` that stores the
         *                      predictions
         */
        virtual std::unique_ptr<PredictionMatrix> predict(const CsrConstView<const float32>& featureMatrix) const = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IPredictor`.
 *
 * @tparam PredictionMatrix The type of the matrix that is used to store the predictions
 */
template<typename PredictionMatrix>
class IPredictorFactory {

    public:

        virtual ~IPredictorFactory() { };

        /**
         * Creates and returns a new object of the type `IPredictor`.
         *
         * @param featureMatrix     A reference to an object of type `CsrConstView` that stores the feature values of
         *                          the query examples to predict for
         * @param model             A reference to an object of type `RuleList` that should be used to obtain
         *                          predictions
         * @param labelVectorSet    A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `IPredictor` that has been created
         */
        virtual std::unique_ptr<IPredictor<PredictionMatrix>> create(
            const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
            const LabelVectorSet* labelVectorSet, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new object of the type `IPredictor`.
         *
         * @param featureMatrix     A reference to an object of type `CsrConstView` that stores the feature values of
         *                          the query examples to predict for
         * @param model             A reference to an object of type `RuleList` that should be used to obtain
         *                          predictions
         * @param labelVectorSet    A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `IPredictor` that has been created
         */
        virtual std::unique_ptr<IPredictor<PredictionMatrix>> create(const CsrConstView<const float32>& featureMatrix,
                                                                     const RuleList& model,
                                                                     const LabelVectorSet* labelVectorSet,
                                                                     uint32 numLabels) const = 0;

};

/**
 * Defines an interface for all classes that allow to configure a predictor.
 *
 * @tparam PredictionMatrix The type of the matrix that is used to store the predictions
 */
template<typename PredictionMatrix>
class IPredictorConfig {

    public:

        virtual ~IPredictorConfig() { };

        /**
         * Creates and returns a new object of type `IPredictorFactory` according to the specified configuration.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples to predict for
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IPredictorFactory` that has been created
         */
        virtual std::unique_ptr<IPredictorFactory<PredictionMatrix>> createPredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

        /**
         * Returns whether the predictor needs access to the label vectors that are encountered in the training data or
         * not.
         *
         * @return True, if the predictor needs access to the label vectors that are encountered in the training data,
         *         false otherwise
         */
        virtual bool isLabelVectorSetNeeded() const = 0;

};
