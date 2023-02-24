/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/prediction_matrix_sparse_binary.hpp"
#include "common/prediction/predictor.hpp"
#include "omp.h"

/**
 * Allows to obtain predictions for multiple query examples by delegating the prediction for individual examples to
 * another class.
 *
 * @tparam T                The type of the predictions
 * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of the
 *                          query examples
 * @tparam Model            The type of the rule-based model that is used to obtain predictions
 */
template<typename T, typename FeatureMatrix, typename Model>
class PredictionDispatcher final {
    public:

        /**
         * Defines an interface for all classes the prediction for individual examples can be delegated to by a
         * `PredictionDispatcher`.
         */
        class IPredictionDelegate {
            public:

                virtual ~IPredictionDelegate() {};

                /**
                 * Obtain predictions for a single query example and stores them in a given prediction matrix.
                 *
                 * @param model             A reference to an object of template type `Model` that should be used to
                 *                          obtain predictions
                 * @param featureMatrix     A reference to an object of template type `FeatureMatrix` that provides
                 *                          row-wise access to the feature values of the query examples
                 * @param predictionMatrix  A reference to an object of type `CContiguousView` that should be used to
                 *                          store the predictions
                 * @param maxRules          The maximum number of rules to be used for prediction or 0, if the number of
                 *                          rules should not be restricted
                 * @param exampleIndex      The index of the query example to predict for
                 */
                virtual void predictForExample(const Model& model, const FeatureMatrix& featureMatrix,
                                               CContiguousView<T>& predictionMatrix, uint32 maxRules,
                                               uint32 exampleIndex) const = 0;
        };

        /**
         * Obtains predictions for multiple query examples and stores them in a given prediction matrix.
         *
         * @param delegate          A reference to an object of type `IPredictionDelegate`, the prediction for
         *                          individual examples should be delegated to
         * @param featureMatrix     A reference to an object of template type `FeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param model             A reference to an object of template type `Model` that should be used to obtain
         *                          predictions
         * @param predictionMatrix  A reference to an object of type `CContiguousView` that should be used to store the
         *                          predictions
         * @param maxRules          The maximum number of rules to be used for prediction or 0, if the number of rules
         *                          should not be restricted
         * @param numThreads        The number of CPU threads to be used to make predictions for different query
         *                          examples in parallel. Must be at least 1
         */
        void predict(const IPredictionDelegate& delegate, const FeatureMatrix& featureMatrix, const Model& model,
                     CContiguousView<T>& predictionMatrix, uint32 maxRules, uint32 numThreads) const {
            uint32 numExamples = featureMatrix.getNumRows();
            const IPredictionDelegate* delegatePtr = &delegate;
            const FeatureMatrix* featureMatrixPtr = &featureMatrix;
            const Model* modelPtr = &model;
            CContiguousView<T>* predictionMatrixPtr = &predictionMatrix;

#pragma omp parallel for firstprivate(numExamples), \
  firstprivate(delegatePtr) firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
    firstprivate(maxRules) schedule(dynamic) num_threads(numThreads)
            for (int64 i = 0; i < numExamples; i++) {
                delegatePtr->predictForExample(*modelPtr, *featureMatrixPtr, *predictionMatrixPtr, maxRules, i);
            }
        }
};

/**
 * An abstract base class for all classes that allow to obtain sparse binary predictions for given query examples.
 *
 * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of the
 *                          query examples
 * @tparam Model            The type of the rule-based model that is used to obtain predictions
 */
template<typename FeatureMatrix, typename Model>
class AbstractBinarySparsePredictor : virtual public IPredictor<BinarySparsePredictionMatrix> {
    private:

        const FeatureMatrix& featureMatrix_;

        const Model& model_;

        uint32 numLabels_;

        uint32 numThreads_;

    protected:

        /**
         * Must be implemented by subclasses in order to obtain predictions for a single query example.
         *
         * @param model             A reference to an object of template type `Model` that should be used to obtain
         *                          predictions
         * @param featureMatrix     A reference to an object of template type `FeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param predictionRow     A reference to an object of type `BinaryLilMatrix::Row` that should be used to store
         *                          the predictions
         * @param numLabels         The number of labels to predict for
         * @param maxRules          The maximum number of rules to be used for prediction or 0, if the number of rules
         *                          should not be restricted
         * @param exampleIndex      The index of the query example to predict for
         */
        virtual void predictForExample(const Model& model, const FeatureMatrix& featureMatrix,
                                       BinaryLilMatrix::row predictionRow, uint32 numLabels, uint32 maxRules,
                                       uint32 exampleIndex) const = 0;

    public:

        /**
         * @param featureMatrix A reference to an object of template type `FeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param model         A reference to an object of template type `Model` that should be used to obtain
         *                      predictions
         * @param numLabels     The number of labels to predict for
         * @param numThreads    The number of CPU threads to be used to make predictions for different query examples in
         *                      parallel. Must be at least 1
         */
        AbstractBinarySparsePredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                      uint32 numThreads)
            : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads) {}

        virtual ~AbstractBinarySparsePredictor() {};

        std::unique_ptr<BinarySparsePredictionMatrix> predict(uint32 maxRules) const override final {
            uint32 numExamples = featureMatrix_.getNumRows();
            uint32 numLabels = numLabels_;
            BinaryLilMatrix predictionMatrix(numExamples);
            const FeatureMatrix* featureMatrixPtr = &featureMatrix_;
            BinaryLilMatrix* predictionMatrixPtr = &predictionMatrix;
            const Model* modelPtr = &model_;
            uint32 numNonZeroElements = 0;

#pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numLabels) \
  firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) firstprivate(maxRules) \
    schedule(dynamic) num_threads(numThreads_)
            for (int64 i = 0; i < numExamples; i++) {
                BinaryLilMatrix::row predictionRow = (*predictionMatrixPtr)[i];
                this->predictForExample(*modelPtr, *featureMatrixPtr, predictionRow, numLabels, maxRules, i);
                numNonZeroElements += (uint32) predictionRow.size();
            }

            return createBinarySparsePredictionMatrix(predictionMatrix, numLabels, numNonZeroElements);
        }
};
