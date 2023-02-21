/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/predictor.hpp"
#include "omp.h"

/**
 * An abstract base class for all classes that allow to obtain dense predictions for given query examples.
 *
 * @tparam T                The type of the predictions
 * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of the
 *                          query examples
 * @tparam Model            The type of the rule-based model that is used to obtain predictions
 */
template<typename T, typename FeatureMatrix, typename Model>
class AbstractPredictor : virtual public IPredictor<DensePredictionMatrix<T>> {
    private:

        const FeatureMatrix& featureMatrix_;

        const Model& model_;

        uint32 numLabels_;

        uint32 numThreads_;

    protected:

        /**
         * Must be implement by subclasses in order to create the matrix that should be used to store predictions.
         *
         * @param model         A reference to an object of template type `Model` that should be used to obtain
         *                      predictions
         * @param numExamples   The number of examples to predict for
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `DensePredictionMatrix` that has been created
         */
        virtual std::unique_ptr<DensePredictionMatrix<T>> createPredictionMatrix(const Model& model, uint32 numExamples,
                                                                                 uint32 numLabels) const = 0;

        /**
         * Must be implemented by subclasses in order to obtain predictions for a single query example.
         *
         * @param model             A reference to an object of template type `Model` that should be used to obtain
         *                          predictions
         * @param featureMatrix     A reference to an object of template type `FeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param predictionMatrix  A reference to an object of type `DensePredictionMatrix` that should be used to
         *                          store the predictions
         * @param maxRules          The maximum number of rules to be used for prediction or 0, if the number of rules
         *                          should not be restricted
         * @param exampleIndex      The index of the query example to predict for
         */
        virtual void predictForExample(const Model& model, const FeatureMatrix& featureMatrix,
                                       DensePredictionMatrix<T>& predictionMatrix, uint32 maxRules,
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
        AbstractPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels, uint32 numThreads)
            : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads) {}

        virtual ~AbstractPredictor() {};

        std::unique_ptr<DensePredictionMatrix<T>> predict(uint32 maxRules) const override final {
            uint32 numExamples = featureMatrix_.getNumRows();
            std::unique_ptr<DensePredictionMatrix<T>> predictionMatrixPtr =
              this->createPredictionMatrix(model_, numExamples, numLabels_);
            const FeatureMatrix* featureMatrixPtr = &featureMatrix_;
            DensePredictionMatrix<T>* predictionMatrixRawPtr = predictionMatrixPtr.get();
            const Model* modelPtr = &model_;

#pragma omp parallel for firstprivate(numExamples) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
  firstprivate(predictionMatrixRawPtr) firstprivate(maxRules) schedule(dynamic) num_threads(numThreads_)
            for (int64 i = 0; i < numExamples; i++) {
                this->predictForExample(*modelPtr, *featureMatrixPtr, *predictionMatrixRawPtr, maxRules, i);
            }

            return predictionMatrixPtr;
        }
};
