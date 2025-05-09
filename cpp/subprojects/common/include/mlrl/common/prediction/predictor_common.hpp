/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/prediction/predictor.hpp"
#include "mlrl/common/util/openmp.hpp"

#include <algorithm>

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
         * Defines an interface for all classes, the prediction for individual examples can be delegated to by a
         * `PredictionDispatcher`.
         */
        class IPredictionDelegate {
            public:

                virtual ~IPredictionDelegate() {}

                /**
                 * Obtains predictions for a single query example.
                 *
                 * @param featureMatrix     A reference to an object of template type `FeatureMatrix` that provides
                 *                          row-wise access to the feature values of the query examples
                 * @param rulesBegin        An iterator of type `Model::const_iterator` to the first rule that should be
                 *                          used for prediction
                 * @param rulesEnd          An iterator of type `Model::const_iterator` to the last rule (exclusive)
                 *                          that should be used for prediction
                 * @param threadIndex       The index of the thread used for prediction
                 * @param exampleIndex      The index of the query example to predict for
                 * @param predictionIndex   The index of the row in the prediction matrix, where the predictions should
                 *                          be stored
                 */
                virtual void predictForExample(const FeatureMatrix& featureMatrix,
                                               typename Model::const_iterator rulesBegin,
                                               typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                               uint32 exampleIndex, uint32 predictionIndex) const = 0;
        };

        /**
         * Obtains predictions for multiple query examples by delegating the prediction for individual examples to a
         * given `PredictionDelegate`.
         *
         * @param delegate                  A reference to an object of type `IPredictionDelegate`, the prediction for
         *                                  individual examples should be delegated to
         * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param rulesBegin                An iterator of type `Model::const_iterator` to the first rule that should be
         *                                  used for prediction
         * @param rulesEnd                  An iterator of type `Model::const_iterator` to the last rule (exclusive)
         *                                  that should be used for prediction
         * @param multiThreadingSettings    An object of type `MultiThreadingSettings` that stores the settings to be
         *                                  used for making predictions for different query examples in parallel
         */
        void predict(const IPredictionDelegate& delegate, const FeatureMatrix& featureMatrix,
                     typename Model::const_iterator rulesBegin, typename Model::const_iterator rulesEnd,
                     MultiThreadingSettings multiThreadingSettings) const {
            uint32 numExamples = featureMatrix.numRows;
            const IPredictionDelegate* delegatePtr = &delegate;
            const FeatureMatrix* featureMatrixPtr = &featureMatrix;

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numExamples) firstprivate(delegatePtr) firstprivate(rulesBegin) \
      firstprivate(rulesEnd) firstprivate(featureMatrixPtr) schedule(dynamic) \
      num_threads(multiThreadingSettings.numThreads)
#endif
            for (int64 i = 0; i < numExamples; i++) {
#if MULTI_THREADING_SUPPORT_ENABLED
                uint32 threadIndex = static_cast<uint32>(omp_get_thread_num());
#else
                uint32 threadIndex = 1;
#endif
                delegatePtr->predictForExample(*featureMatrixPtr, rulesBegin, rulesEnd, threadIndex, i, i);
            }
        }
};

/**
 * Allows to obtain sparse binary predictions for multiple query examples by delegating the prediction for individual
 * examples to another class.
 *
 * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of the
 *                          query examples
 * @tparam Model            The type of the rule-based model that is used to obtain predictions
 */
template<typename FeatureMatrix, typename Model>
class BinarySparsePredictionDispatcher final {
    public:

        /**
         * Defines an interface for all classes, the prediction for individual examples can be delegated to by a
         * `BinarySparsePredictionDispatcher`.
         */
        class IPredictionDelegate {
            public:

                virtual ~IPredictionDelegate() {}

                /**
                 * Obtains predictions for a single query example.
                 *
                 * @param featureMatrix     A reference to an object of template type `FeatureMatrix` that provides
                 *                          row-wise access to the feature values of the query examples
                 * @param rulesBegin        An iterator of type `Model::const_iterator` to the first rule that should be
                 *                          used for prediction
                 * @param rulesEnd          An iterator of type `Model::const_iterator` to the last rule (exclusive)
                 *                          that should be used for prediction
                 * @param threadIndex       The index of the thread used for prediction
                 * @param exampleIndex      The index of the query example to predict for
                 * @param predictionIndex   The index of the row in the prediction matrix, where the predictions should
                 *                          be stored
                 * @return                  The number of dense elements explicitly stored in the affected row of the
                                            prediction matrix
                 */
                virtual uint32 predictForExample(const FeatureMatrix& featureMatrix,
                                                 typename Model::const_iterator rulesBegin,
                                                 typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                                 uint32 exampleIndex, uint32 predictionIndex) const = 0;
        };

        /**
         * Obtains predictions for multiple query examples by delegating the prediction for individual examples to a
         * given `IPredictionDelegate`.
         *
         * @param delegate                  A reference to an object of type `IPredictionDelegate`, the prediction for
         *                                  individual examples should be delegated to
         * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param rulesBegin                An iterator of type `Model::const_iterator` to the first rule that should be
         *                                  used for prediction
         * @param rulesEnd                  An iterator of type `Model::const_iterator` to the last rule (exclusive)
         *                                  that should be used for prediction
         * @param multiThreadingSettings    An object of type `MultiThreadingSettings` that stores the settings to be
         *                                  used for making predictions for different query examples in parallel
         * @return                          The total number of dense predictions explicitly stored in the affected row
                                            of the prediction matrix
         */
        uint32 predict(const IPredictionDelegate& delegate, const FeatureMatrix& featureMatrix,
                       typename Model::const_iterator rulesBegin, typename Model::const_iterator rulesEnd,
                       MultiThreadingSettings multiThreadingSettings) const {
            uint32 numExamples = featureMatrix.numRows;
            const IPredictionDelegate* delegatePtr = &delegate;
            const FeatureMatrix* featureMatrixPtr = &featureMatrix;
            uint32 numDenseElements = 0;

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for reduction(+ : numDenseElements) firstprivate(numExamples) firstprivate(delegatePtr) \
      firstprivate(rulesBegin) firstprivate(rulesEnd) firstprivate(featureMatrixPtr) schedule(dynamic) \
      num_threads(multiThreadingSettings.numThreads)
#endif
            for (int64 i = 0; i < numExamples; i++) {
#if MULTI_THREADING_SUPPORT_ENABLED
                uint32 threadIndex = static_cast<uint32>(omp_get_thread_num());
#else
                uint32 threadIndex = 1;
#endif
                numDenseElements +=
                  delegatePtr->predictForExample(*featureMatrixPtr, rulesBegin, rulesEnd, threadIndex, i, i);
            }

            return numDenseElements;
        }
};

/**
 * An abstract base class for all implementations of the class `IIncrementalPredictor`.
 *
 * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of the
 *                          query examples
 * @tparam Model            The type of the rule-based model that is used to obtain predictions
 * @tparam PredictionMatrix The type of the matrix that is used to store the predictions
 */
template<typename FeatureMatrix, typename Model, typename PredictionMatrix>
class AbstractIncrementalPredictor : public IIncrementalPredictor<PredictionMatrix> {
    private:

        const FeatureMatrix& featureMatrix_;

        const MultiThreadingSettings multiThreadingSettings_;

        typename Model::const_iterator current_;

        typename Model::const_iterator end_;

    protected:

        /**
         * Must be implemented by subclasses in order to obtain predictions.
         *
         * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param multiThreadingSettings    An object of type `MultiThreadingSettings` that stores the settings to be
         *                                  used for making predictions for different query examples in parallel
         * @param rulesBegin                An iterator of type `Model::const_iterator` to the first rule that should be
         *                                  used for prediction
         * @param rulesEnd                  An iterator of type `Model::const_iterator` to the last rule (exclusive)
         *                                  that should be used for prediction
         * @return                          A reference to an object of template type `PredictionMatrix` that stores the
         *                                  predictions
         */
        virtual PredictionMatrix& applyNext(const FeatureMatrix& featureMatrix,
                                            MultiThreadingSettings multiThreadingSettings,
                                            typename Model::const_iterator rulesBegin,
                                            typename Model::const_iterator rulesEnd) = 0;

    public:

        /**
         * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param model                     A reference to an object of template type `Model` that should be used for
         *                                  prediction
         * @param multiThreadingSettings    An object of type `MultiThreadingSettings` that stores the settings to be
         *                                  used for making predictions for different query examples in parallel
         * @param maxRules                  The maximum number of rules to be used for prediction. Must be at least 1 or
         *                                  0, if the number of rules should not be restricted
         */
        AbstractIncrementalPredictor(const FeatureMatrix& featureMatrix, const Model& model,
                                     MultiThreadingSettings multiThreadingSettings, uint32 maxRules)
            : featureMatrix_(featureMatrix), multiThreadingSettings_(multiThreadingSettings),
              current_(model.used_cbegin(maxRules)), end_(model.used_cend(maxRules)) {}

        virtual ~AbstractIncrementalPredictor() override {}

        uint32 getNumNext() const override final {
            return static_cast<uint32>(end_ - current_);
        }

        PredictionMatrix& applyNext(uint32 stepSize) override final {
            typename Model::const_iterator next = current_ + std::min(stepSize, this->getNumNext());
            PredictionMatrix& predictionMatrix =
              this->applyNext(featureMatrix_, multiThreadingSettings_, current_, next);
            current_ = next;
            return predictionMatrix;
        }
};
