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
 * Defines an interface for all classes that allow to obtain predictions for given query examples incrementally.
 *
 * @tparam PredictionMatrix The type of the matrix that is used to store the predictions
 */
template<typename PredictionMatrix>
class IIncrementalPredictor {
    public:

        virtual ~IIncrementalPredictor() {};

        /**
         * Returns whether there are any remaining ensemble members that have not been used yet or not.
         *
         * @return True, if there are any remaining ensemble members, false otherwise
         */
        virtual bool hasNext() const {
            return this->getNumNext() > 0;
        }

        /**
         * Returns the number of remaining ensemble members that have not been used yet.
         *
         * @return The number of remaining ensemble members
         */
        virtual uint32 getNumNext() const = 0;

        /**
         * Updates the current predictions by considering several of the remaining ensemble members. If not enough
         * ensemble members are remaining, only the available ones will be used for updating the current predictions.
         *
         * @param stepSize  The number of additional ensemble members to be considered for prediction
         * @return          A reference to an object of template type `PredictionMatrix` that stores the updated
         *                  predictions
         */
        virtual PredictionMatrix& applyNext(uint32 stepSize) = 0;
};

/**
 * Defines an interface for all classes that allow to make prediction for given query examples.
 *
 * @tparam PredictionMatrix The type of the matrix that is used to store the predictions
 */
template<typename PredictionMatrix>
class IPredictor {
    public:

        virtual ~IPredictor() {};

        /**
         * Obtains and returns predictions for all query examples.
         *
         * @return An unique pointer to an object of template type `PredictionMatrix` that stores the predictions
         */
        virtual std::unique_ptr<PredictionMatrix> predict() const = 0;

        /**
         * Returns whether the predictor allows to obtain predictions incrementally or not.
         *
         * @return True, if the predictor allows to obtain predictions incrementally, false otherwise
         */
        virtual bool canPredictIncrementally() const = 0;

        /**
         * Creates and returns a predictor that may be used to obtain predictions incrementally. If incremental
         * prediction is not supported, a `std::runtime_error` is thrown.
         *
         * @throws std::runtime_exception   The exception that is thrown if incremental prediction is not supported
         * @return                          An unique pointer to an object of type `IIncrementalPredictor` that may be
         *                                  used to obtain predictions incrementally
         */
        virtual std::unique_ptr<IIncrementalPredictor<PredictionMatrix>> createIncrementalPredictor() const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IPredictor`.
 *
 * @tparam Predictor The type of the instances that are created by the factory
 */
template<typename Predictor>
class IPredictorFactory {
    public:

        virtual ~IPredictorFactory() {};

        /**
         * Creates and returns a new object of the template type `Predictor`.
         *
         * @param featureMatrix     A reference to an object of type `CsrConstView` that stores the feature values of
         *                          the query examples to predict for
         * @param model             A reference to an object of type `RuleList` that should be used to obtain
         *                          predictions
         * @param labelVectorSet    A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of template type `Predictor` that has been created
         */
        virtual std::unique_ptr<Predictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                  const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                  uint32 numLabels) const = 0;

        /**
         * Creates and returns a new object of the template type `Predictor`.
         *
         * @param featureMatrix     A reference to an object of type `CsrConstView` that stores the feature values of
         *                          the query examples to predict for
         * @param model             A reference to an object of type `RuleList` that should be used to obtain
         *                          predictions
         * @param labelVectorSet    A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of template type `Predictor` that has been created
         */
        virtual std::unique_ptr<Predictor> create(const CsrConstView<const float32>& featureMatrix,
                                                  const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                  uint32 numLabels) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a predictor.
 *
 * @tparam PredictorFactory The type of the factory that allows to create instances of the predictor
 */
template<typename PredictorFactory>
class IPredictorConfig {
    public:

        virtual ~IPredictorConfig() {};

        /**
         * Creates and returns a new object of type `IPredictorFactory` according to the specified configuration.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples to predict for
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of template type `PredictorFactory` that has been created
         */
        virtual std::unique_ptr<PredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                         uint32 numLabels) const = 0;

        /**
         * Returns whether the predictor needs access to the label vectors that are encountered in the training data or
         * not.
         *
         * @return True, if the predictor needs access to the label vectors that are encountered in the training data,
         *         false otherwise
         */
        virtual bool isLabelVectorSetNeeded() const = 0;
};
